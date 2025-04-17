import socket
import threading
import time
import numpy as np
import struct
import ImageStreamIOWrap as ISIO
import argparse
import os
import logging
import coloredlogs  # For colored logs

# Setup logger
logger = logging.getLogger(__name__)

class ImageStreamIO:
    @staticmethod
    def open_shm(shm_name):
        """
        Open an existing shared memory image stream.
        Returns the image object if successful, or None if the open fails.
        """
        img = ISIO.Image()
        ret = img.open(shm_name)
        if ret != 0:
            return None
        return img

    @staticmethod
    def read_shm(shm_obj):
        """
        Convert the shared memory image to a NumPy array.
        Ensures the result is a numpy array and removes any singleton dimensions.
        """
        data = shm_obj.copy()
        # Ensure it's a numpy array
        if not isinstance(data, np.ndarray):
            data = np.array(data)
        # Squeeze out any dimensions that are of size 1
        return np.squeeze(data)

    @staticmethod
    def get_data_size(shm_obj):
        """
        Calculate the size in bytes of the shared memory image.
        Assumes the image data is of type FLOAT (np.float32).
        """
        # Check if metadata or size attribute exists and is valid
        if not hasattr(shm_obj, 'md') or not hasattr(shm_obj.md, 'size') or not shm_obj.md.size:
            logger.error("SHM object metadata or size is invalid.")
            return 0
            
        shape = shm_obj.md.size
        # Ensure shape is iterable and contains numbers
        if not hasattr(shape, '__iter__') or not all(isinstance(dim, (int, float)) for dim in shape):
            logger.error(f"SHM object shape is invalid: {shape}")
            return 0

        # Ensure dtype is valid before calculating itemsize
        try:
            dtype = np.dtype(np.float32)
            itemsize = dtype.itemsize
        except TypeError:
            logger.error("Invalid data type specified for item size calculation.")
            return 0
            
        total_elements = np.prod(shape)
        return total_elements * itemsize

# Bandwidth manager class
class BandwidthManager:
    def __init__(self, max_bandwidth_bps):
        self.max_bandwidth = max_bandwidth_bps
        self.start_time = time.time()
        self.bytes_sent = 0

    def update(self, bytes_to_send):
        """Update bandwidth usage and return required sleep time"""
        self.bytes_sent += bytes_to_send
        elapsed = time.time() - self.start_time
        target_time = self.bytes_sent / self.max_bandwidth
        return max(0, target_time - elapsed)

    def reset(self):
        self.start_time = time.time()
        self.bytes_sent = 0

def handle_client(conn, addr, bw_limit_bps):
    """
    Handle client connection:
    1. Receive requested shared memory name
    2. Verify shared memory exists
    3. Stream frames with bandwidth management
    """
    logger.info(f"Starting handler for client {addr}")
    try:
        import ImageStreamIOWrap as ISIO
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

        frame_size = ImageStreamIO.get_data_size(shm_obj)
        if frame_size == 0:
            logger.error(f"Calculated frame size is 0 for SHM {shm_name}. Aborting client {addr}.")
            try:
                conn.send(struct.pack('!Q', 0))  # Indicate error or invalid size
            except (BrokenPipeError, ConnectionResetError):
                logger.warning(f"Client {addr} disconnected before frame size error response.")
            return

        conn.send(struct.pack('!Q', frame_size))  # Send valid frame size

        # Send the image shape
        shape = shm_obj.md.size
        shape = tuple(dim for dim in shape if dim > 1)
        if len(shape) == 1:
            shape = (shape[0], 1)
        elif not shape:
            shape = (1, 1)
            logger.warning(f"SHM {shm_name} resulted in an empty shape after filtering. Using (1, 1).")

        if len(shape) != 2:
            logger.error(f"Final shape {shape} for SHM {shm_name} is not 2D. Cannot send shape to client {addr}.")
            return

        conn.send(struct.pack('!II', *shape))
        logger.debug(f"Sent shape {shape} to client {addr}")

        bw_manager = BandwidthManager(bw_limit_bps)
        logger.info(f"Bandwidth limit for {addr} set to {bw_limit_bps / (1024*1024):.2f} MB/s")

        running = True

        while running:
            try:
                frame = ImageStreamIO.read_shm(shm_obj)
                if frame is None or frame.size == 0:
                    logger.warning(f"Failed to read valid frame from SHM {shm_name} for client {addr}. Skipping frame.")
                    time.sleep(0.01)
                    continue

                serialized = frame.tobytes()
                data_len = len(serialized)

                header = struct.pack('!Q', data_len)
                conn.sendall(header + serialized)
                logger.debug(f"Sent frame of size {data_len} bytes to {addr}")

                sleep_time = bw_manager.update(data_len)
                if sleep_time > 0:
                    logger.debug(f"Sleeping for {sleep_time:.4f}s to manage bandwidth for {addr}")
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
        try:
            conn.close()
        except Exception as e:
            logger.error(f"Error closing connection for {addr}: {str(e)}")
        logger.info(f"Handler for client {addr} terminated")

def start_server(host='127.0.0.1', port=5123, max_clients=None, bw_limit_mbps=10.0):
    """Start streaming server with bandwidth management and client limit"""
    bw_limit_bps = bw_limit_mbps * 1024 * 1024

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            s.bind((host, port))
            s.listen()
            logger.info(f"Server listening on {host}:{port}")
            if max_clients:
                logger.info(f"Maximum clients allowed: {max_clients}")
            else:
                logger.info("No client limit set.")
            logger.info(f"Default bandwidth limit per client: {bw_limit_mbps} MB/s")

            client_threads = []

            while True:
                try:
                    client_threads = [t for t in client_threads if t.is_alive()]
                    active_client_count = len(client_threads)
                    logger.debug(f"Active clients: {active_client_count}")

                    if max_clients is not None and active_client_count >= max_clients:
                        logger.warning(f"Max client limit ({max_clients}) reached. Waiting for a slot...")
                        time.sleep(1)
                        continue

                    conn, addr = s.accept()
                    logger.info(f"New connection from {addr}")
                    client_thread = threading.Thread(
                        target=handle_client,
                        args=(conn, addr, bw_limit_bps),
                        daemon=True
                    )
                    client_thread.start()
                    client_threads.append(client_thread)

                except KeyboardInterrupt:
                    logger.info("Shutdown signal received. Closing server.")
                    break
                except Exception as e:
                    logger.error(f"Error accepting connection: {str(e)}", exc_info=True)
                    time.sleep(1)

        except Exception as e:
            logger.critical(f"Server error: {str(e)}", exc_info=True)
        finally:
            logger.info("Shutting down server.")

def pin_to_cores(cores):
    """
    Pin the current process to specific CPU cores.

    Args:
        cores: A list of core numbers to pin the process to
    """
    try:
        import psutil
        p = psutil.Process()
        p.cpu_affinity(cores)
        logger.info(f"Process pinned to CPU cores: {cores}")
    except ImportError:
        logger.warning("psutil module not available. CPU pinning not supported.")
    except AttributeError:
        logger.warning("CPU affinity setting not available on this system/psutil version.")
    except Exception as e:
        logger.error(f"Failed to pin process to cores {cores}: {str(e)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='SHM Viewer Server')
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

    start_server(host=args.ip, port=args.port, max_clients=args.max_clients, bw_limit_mbps=args.bw_limit)