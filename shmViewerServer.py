import socket
import threading
import time
import numpy as np
import struct
import argparse
import os
import logging
import coloredlogs  # For colored logs
import math  # For isnan/isinf checks

from isio_wrapper import ImageStreamIO
from server_utils import BandwidthManager, ServerBase, pin_to_cores, recv_all

# Setup logger
logger = logging.getLogger(__name__)

class ShmStreamingServer(ServerBase):
    """
    Server that streams latest SHM data to clients.
    """
    def handle_client(self, conn, addr, bw_limit_bps):
        """
        Handle client connection:
        1. Receive requested shared memory name
        2. Verify shared memory exists
        3. Stream frames (including count and timestamp) with bandwidth management
        4. Attempt to re-initialize SHM connection if data appears invalid.
        """
        logger.info(f"Starting streaming handler for client {addr}")
        shm_obj = None  # Initialize shm_obj to None
        shm_name = None  # Initialize shm_name
        try:
            # Set a timeout for receiving the SHM name to prevent hanging
            conn.settimeout(10.0)

            # Negotiate shared memory name
            data = conn.recv(256)
            if not data:
                logger.warning(f"Client {addr} disconnected before sending SHM name")
                return

            shm_name = data.decode().strip('\x00')
            logger.info(f"Client {addr} requested SHM: {shm_name}")

            # Reset timeout for streaming
            conn.settimeout(None)

            shm_obj = ImageStreamIO.open_shm(shm_name)

            if not shm_obj:
                logger.warning(f"SHM {shm_name} not found for client {addr}")
                try:
                    conn.send(struct.pack('!Q', 0))  # Send invalid size
                except (BrokenPipeError, ConnectionResetError):
                    logger.warning(f"Client {addr} disconnected before SHM existence check response.")
                return

            # Initial metadata check and sending
            frame_size = ImageStreamIO.get_data_size(shm_obj)
            if frame_size == 0:
                logger.error(f"Initial calculated frame size is 0 for SHM {shm_name}. Aborting client {addr}.")
                try:
                    conn.send(struct.pack('!Q', 0))  # Indicate error or invalid size
                except (BrokenPipeError, ConnectionResetError):
                    logger.warning(f"Client {addr} disconnected before frame size error response.")
                return

            conn.send(struct.pack('!Q', frame_size))  # Send valid frame size

            shape = ImageStreamIO.get_shape(shm_obj)
            if len(shape) != 2:
                logger.error(f"Initial shape {shape} for SHM {shm_name} is not 2D. Cannot send shape to client {addr}.")
            else:
                try:
                    conn.send(struct.pack('!II', *shape))
                    logger.debug(f"Sent initial shape {shape} to client {addr}")
                except (BrokenPipeError, ConnectionResetError):
                    logger.warning(f"Client {addr} disconnected before initial shape could be sent.")
                    return

            bw_manager = BandwidthManager(bw_limit_bps)
            logger.info(f"Bandwidth limit for {addr} set to {bw_limit_bps / (1024*1024):.2f} MB/s")

            running = True
            consecutive_read_failures = 0
            max_consecutive_read_failures = 5  # Threshold to trigger re-init
            invalid_data_threshold_fraction = 0.25  # Allow up to 25% NaN/Inf

            # Define struct formats for count (int64) and time (float64)
            count_time_format = '!qd' # Signed long long (int64), double (float64)
            count_time_size = struct.calcsize(count_time_format)

            while running:
                try:
                    frame, frame_number, frame_time = ImageStreamIO.read_shm(shm_obj) # frame_time is datetime

                    # Check for invalid frame data
                    invalid_data = False
                    if frame is None or frame.size == 0:
                        logger.warning(f"Read empty frame from SHM {shm_name} for client {addr}.")
                        invalid_data = True
                    # Check for excessive NaN/Inf
                    elif np.issubdtype(frame.dtype, np.floating):
                        nan_count = np.isnan(frame).sum()
                        inf_count = np.isinf(frame).sum()
                        total_invalid_count = nan_count + inf_count
                        threshold = frame.size * invalid_data_threshold_fraction

                        if total_invalid_count > threshold:
                            logger.warning(f"Detected {total_invalid_count} NaN/Inf values (>{threshold:.0f}, {invalid_data_threshold_fraction*100:.1f}%) in frame from SHM {shm_name} for client {addr}.")
                            invalid_data = True
                        elif total_invalid_count > 0:
                            # Log if there are *some* invalid values, but below threshold
                            logger.debug(f"Detected {total_invalid_count} NaN/Inf values (below threshold) in frame from SHM {shm_name} for client {addr}.")
                            # Do not set invalid_data = True here, tolerate these values

                    if invalid_data:
                        consecutive_read_failures += 1
                        logger.warning(f"Consecutive read failures/excessive invalid data count: {consecutive_read_failures} for {addr}")
                        if consecutive_read_failures >= max_consecutive_read_failures:
                            logger.warning(f"Threshold reached. Attempting to re-initialize SHM '{shm_name}' for {addr}.")

                            # Attempt to close the existing object
                            try:
                                if hasattr(shm_obj, 'close') and callable(shm_obj.close):
                                    shm_obj.close()
                                    logger.info(f"Closed potentially stale SHM object for {addr}.")
                                else:
                                    del shm_obj
                                    logger.info(f"Deleted potentially stale SHM object reference for {addr}.")
                                    shm_obj = None
                            except Exception as close_err:
                                logger.error(f"Error closing SHM object for {addr}: {close_err}", exc_info=True)

                            # Attempt to reopen
                            shm_obj = ImageStreamIO.open_shm(shm_name)
                            if not shm_obj:
                                logger.error(f"Failed to re-open SHM {shm_name} for client {addr} after detecting issues. Terminating handler.")
                                running = False
                                break

                            # Re-check metadata after reopening
                            new_frame_size = ImageStreamIO.get_data_size(shm_obj)
                            if new_frame_size == 0:
                                logger.error(f"Re-opened SHM {shm_name} has zero size for client {addr}. Terminating handler.")
                                running = False
                                break

                            logger.info(f"Successfully re-opened SHM {shm_name} for client {addr}. New size: {new_frame_size} bytes.")
                            frame_size = new_frame_size
                            consecutive_read_failures = 0
                            continue

                        else:
                            time.sleep(0.05)
                            continue  # Skip sending this frame deemed invalid

                    # If we reach here, frame is considered valid enough to send
                    consecutive_read_failures = 0  # Reset counter on successful valid read

                    serialized_frame = frame.tobytes()
                    frame_data_len = len(serialized_frame)

                    # Convert datetime to float timestamp before packing
                    timestamp_float = frame_time.timestamp()

                    # Pack count and float timestamp
                    packed_count_time = struct.pack(count_time_format, frame_number, timestamp_float)

                    # Calculate total payload length
                    total_payload_len = count_time_size + frame_data_len

                    # Pack header with total payload length
                    header = struct.pack('!Q', total_payload_len)

                    # Send header, then packed count/time, then frame data
                    conn.sendall(header + packed_count_time + serialized_frame)

                    # Update bandwidth manager with total bytes sent (header + payload)
                    bytes_sent_this_frame = 8 + total_payload_len
                    sleep_time = bw_manager.update(bytes_sent_this_frame)
                    if sleep_time > 0:
                        time.sleep(sleep_time)

                    if time.time() - bw_manager.start_time > 1:
                        bw_manager.reset()

                except (ConnectionResetError, BrokenPipeError, ConnectionAbortedError) as e:
                    logger.warning(f"Client {addr} connection error: {str(e)}")
                    running = False
                    break
                except Exception as e:
                    logger.error(f"Unexpected error in client loop for {addr}: {str(e)}", exc_info=True)
                    running = False
                    break

        except socket.timeout:
            logger.warning(f"Client {addr} timed out waiting for SHM name.")
        except Exception as e:
            logger.error(f"Client {addr} disconnected with error: {str(e)}", exc_info=True)
        finally:
            if shm_obj:
                try:
                    if hasattr(shm_obj, 'close') and callable(shm_obj.close):
                        shm_obj.close()
                        logger.info(f"Closed SHM object for {addr} during final cleanup.")
                    else:
                        del shm_obj
                except Exception as close_err:
                    logger.error(f"Error closing SHM object during final cleanup for {addr}: {close_err}")

            try:
                conn.close()
            except Exception as e:
                logger.error(f"Error closing connection for {addr}: {str(e)}")
            logger.info(f"Handler for client {addr} terminated")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='SHM Viewer Streaming Server')
    parser.add_argument('--ip', type=str, default='127.0.0.1',
                        help='IP address to bind the server to (default: 127.0.0.1)')
    parser.add_argument('--port', type=int, default=5123,
                        help='Port to bind the server to (default: 5123)')
    parser.add_argument('--cores', type=str, default=None,
                        help='CPU core(s) to pin this process to (e.g., "0,1,2" or "3")')
    parser.add_argument('--bw-limit', type=float, default=10.0,
                        help='Maximum bandwidth per client in MB/s (default: 10.0)')
    parser.add_argument('--max-clients', type=int, default=None,
                        help='Maximum number of concurrent clients (default: unlimited)')
    parser.add_argument('--log-level', type=str, default='INFO',
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                        help='Set the logging level (default: INFO)')

    args = parser.parse_args()

    log_level = getattr(logging, args.log_level.upper(), logging.INFO)
    coloredlogs.install(
        level=log_level,
        logger=logger,
        fmt='%(asctime)s [%(levelname)s] %(name)s:%(funcName)s - %(message)s',
        level_styles={'debug': {'color': 'green'}, 'info': {'color': 'cyan'}, 'warning': {'color': 'yellow'}, 'error': {'color': 'red'}, 'critical': {'bold': True, 'color': 'red'}}
    )

    if args.cores:
        try:
            cores = [int(core.strip()) for core in args.cores.split(',')]
            pin_to_cores(cores)
        except ValueError:
            logger.error(f"Invalid core specification: {args.cores}. Format should be comma-separated integers.")
        except Exception as e:
            logger.error(f"Error setting up CPU pinning: {e}")
    
    server = ShmStreamingServer(
        host=args.ip, 
        port=args.port, 
        max_clients=args.max_clients, 
        bw_limit_mbps=args.bw_limit
    )
    server.start()