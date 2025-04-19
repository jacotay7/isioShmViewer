#!/usr/bin/env python3
import socket
import threading
import time
import numpy as np
import struct
import argparse
import os
import logging
import coloredlogs
import collections
from datetime import datetime

from isio_wrapper import ImageStreamIO
from server_utils import BandwidthManager, ServerBase, pin_to_cores, recv_all

# Setup logger
logger = logging.getLogger(__name__)

class FrameBuffer:
    """
    A buffer that stores frames from shared memory with their metadata.
    Keeps track of frame counts and timestamps for data consistency.
    """
    def __init__(self, max_frames=100):
        self.max_frames = max_frames
        self.buffer = collections.deque(maxlen=max_frames)
        self.frame_count_to_index = {}  # Maps frame counts to buffer indices
        self.last_frame_count = None
        self.lock = threading.Lock()
    
    def add_frame(self, frame, frame_count, timestamp):
        """
        Add a frame to the buffer.
        
        Args:
            frame (np.ndarray): The frame data
            frame_count (int): Frame counter value
            timestamp (float): Frame timestamp
        """
        with self.lock:
            # Add frame to buffer
            self.buffer.append((frame.copy(), frame_count, timestamp))
            
            # Update the mapping - note that buffer indices change when items are removed
            self._update_index_mapping()
            
            # Update the last frame count
            self.last_frame_count = frame_count
    
    def _update_index_mapping(self):
        """Update the mapping of frame counts to buffer indices."""
        self.frame_count_to_index.clear()
        for idx, (_, frame_count, _) in enumerate(self.buffer):
            self.frame_count_to_index[frame_count] = idx
    
    def get_frame_range(self, start_frame, end_frame=None, max_frames=None):
        """
        Get a range of frames from the buffer.
        
        Args:
            start_frame (int): Starting frame count
            end_frame (int, optional): Ending frame count (inclusive). If None, uses last frame.
            max_frames (int, optional): Maximum number of frames to return
            
        Returns:
            list: List of (frame, count, timestamp) tuples
        """
        with self.lock:
            if not self.buffer:
                return []
            
            # If end_frame is not specified, use the last frame
            if end_frame is None:
                end_frame = self.last_frame_count
            
            # Find frames within the specified range
            result = []
            for frame, count, timestamp in self.buffer:
                if start_frame <= count <= end_frame:
                    result.append((frame, count, timestamp))
                    
                    # If max_frames is specified and we've reached that limit, stop adding frames
                    if max_frames is not None and len(result) >= max_frames:
                        break
            
            return result
    
    def get_newest_frames(self, num_frames):
        """
        Get the most recent frames from the buffer.
        
        Args:
            num_frames (int): Number of frames to retrieve
            
        Returns:
            list: List of (frame, count, timestamp) tuples
        """
        with self.lock:
            count = min(num_frames, len(self.buffer))
            return list(self.buffer)[-count:]
    
    def get_frame_counts(self):
        """
        Get the range of frame counts currently in the buffer.
        
        Returns:
            tuple: (min_count, max_count) or (None, None) if buffer is empty
        """
        with self.lock:
            if not self.buffer:
                return None, None
            min_count = min(count for _, count, _ in self.buffer)
            max_count = max(count for _, count, _ in self.buffer)
            return min_count, max_count

    def get_size(self):
        """
        Get the current number of frames in the buffer.
        
        Returns:
            int: Number of frames
        """
        with self.lock:
            return len(self.buffer)


class ShmBufferingServer(ServerBase):
    """
    Server that buffers SHM data and sends contiguous blocks to clients.
    """
    def __init__(self, host='127.0.0.1', port=5124, max_clients=None, 
                 bw_limit_mbps=10.0, buffer_size=100, poll_interval=0.01, logger=None):
        super().__init__(host, port, max_clients, bw_limit_mbps, logger)
        self.buffers = {}  # Dictionary mapping SHM names to their frame buffers
        self.buffer_size = buffer_size
        self.poll_interval = poll_interval
        self.monitors = {}  # Dictionary mapping SHM names to their monitor threads
        self.monitor_lock = threading.Lock()
    
    def _monitor_shm(self, shm_name):
        """
        Monitor a shared memory object, updating the buffer when new frames arrive.
        
        Args:
            shm_name (str): Name of the shared memory to monitor
        """
        logger.info(f"Starting monitor for SHM: {shm_name}")
        
        shm_obj = None
        last_frame_count = None
        consecutive_failures = 0
        max_consecutive_failures = 5
        
        # Create buffer if it doesn't exist
        with self.monitor_lock:
            if shm_name not in self.buffers:
                self.buffers[shm_name] = FrameBuffer(max_frames=self.buffer_size)
        
        try:
            while self.running:
                try:
                    # (Re)open SHM if needed
                    if shm_obj is None:
                        shm_obj = ImageStreamIO.open_shm(shm_name)
                        if not shm_obj:
                            logger.warning(f"SHM {shm_name} not found, retrying in 1 second...")
                            time.sleep(1)
                            continue
                        logger.info(f"Opened SHM: {shm_name}")
                    
                    # Read frame
                    frame, frame_count, frame_time = ImageStreamIO.read_shm(shm_obj)
                    
                    # Check if frame is valid
                    if frame is None or frame.size == 0:
                        consecutive_failures += 1
                        logger.warning(f"Invalid frame from SHM: {shm_name}, attempt {consecutive_failures}/{max_consecutive_failures}")
                        if consecutive_failures >= max_consecutive_failures:
                            logger.warning(f"Too many failures, attempting to reopen SHM: {shm_name}")
                            shm_obj = None  # Force reopening in next iteration
                            consecutive_failures = 0
                        if self.poll_interval > 0:
                            time.sleep(self.poll_interval)
                        else:
                            time.sleep(0.000001)  # Minimal sleep to yield CPU if needed
                        continue
                    
                    # Reset failure counter on success
                    consecutive_failures = 0
                    
                    # Check for actual frame count discontinuity (gaps or unexpected jumps)
                    # Ignore the case where frame_count == last_frame_count (reading the same frame again)
                    if last_frame_count is not None and frame_count != last_frame_count and frame_count != last_frame_count + 1:
                        logger.debug(f"Frame count discontinuity detected in {shm_name}: last={last_frame_count}, current={frame_count}")

                    # Check if this is a genuinely new frame (count is greater than the last one seen)
                    if last_frame_count is None or frame_count > last_frame_count:
                        # Convert datetime to timestamp
                        timestamp = frame_time.timestamp()
                        
                        # Add to buffer
                        self.buffers[shm_name].add_frame(frame, frame_count, timestamp)
                        last_frame_count = frame_count
                        
                        # Debug logging
                        if frame_count % 1000 == 0:  # Log every 100 frames to avoid spamming
                            min_count, max_count = self.buffers[shm_name].get_frame_counts()
                            logger.debug(f"Buffer for {shm_name}: size={self.buffers[shm_name].get_size()}, frames={min_count}-{max_count}")
                    
                    # Sleep only if a positive poll interval is set
                    if self.poll_interval > 0:
                        time.sleep(self.poll_interval)
                    
                except Exception as e:
                    logger.error(f"Error monitoring SHM {shm_name}: {str(e)}", exc_info=True)
                    consecutive_failures += 1
                    if consecutive_failures >= max_consecutive_failures:
                        logger.warning(f"Too many errors, attempting to reopen SHM: {shm_name}")
                        shm_obj = None  # Force reopening in next iteration
                        consecutive_failures = 0
                    if self.poll_interval > 0:
                        time.sleep(self.poll_interval)
                    else:
                        time.sleep(0.001)  # Slightly longer sleep on error in high-perf mode
                    
        finally:
            # Clean up
            if shm_obj:
                try:
                    if hasattr(shm_obj, 'close') and callable(shm_obj.close):
                        shm_obj.close()
                        logger.info(f"Closed SHM: {shm_name}")
                except Exception as e:
                    logger.error(f"Error closing SHM {shm_name}: {str(e)}")
            
            # Remove buffer if server is shutting down
            if not self.running:
                with self.monitor_lock:
                    if shm_name in self.buffers:
                        del self.buffers[shm_name]
                    if shm_name in self.monitors:
                        del self.monitors[shm_name]
                logger.info(f"Monitor for SHM {shm_name} stopped and resources cleaned up")
    
    def _ensure_monitor_running(self, shm_name):
        """
        Ensure a monitor thread is running for the specified SHM.
        
        Args:
            shm_name (str): Name of the shared memory to monitor
            
        Returns:
            bool: True if monitor is (now) running, False otherwise
        """
        with self.monitor_lock:
            # Check if monitor already exists
            if shm_name in self.monitors and self.monitors[shm_name].is_alive():
                return True
            
            # Start a new monitor thread
            monitor = threading.Thread(
                target=self._monitor_shm,
                args=(shm_name,),
                daemon=True
            )
            self.monitors[shm_name] = monitor
            monitor.start()
            return True
    
    def handle_client(self, conn, addr, bw_limit_bps):
        """
        Handle client connection for the buffering server.
        
        Args:
            conn (socket): Client connection socket
            addr (tuple): Client address
            bw_limit_bps (float): Bandwidth limit in bytes per second
        """
        logger.info(f"Starting buffering handler for client {addr}")
        
        try:
            # Set a timeout for receiving the initial handshake
            conn.settimeout(10.0)
            
            # Receive SHM name (padded to 256 bytes)
            data = conn.recv(256)
            if not data:
                logger.warning(f"Client {addr} disconnected before sending SHM name")
                return
                
            shm_name = data.decode().strip('\x00')
            logger.info(f"Client {addr} requested SHM: {shm_name}")
            
            # Make sure monitor is running for this SHM
            if not self._ensure_monitor_running(shm_name):
                logger.warning(f"Failed to start monitor for SHM {shm_name}, rejecting client {addr}")
                try:
                    # Send failure indication
                    conn.send(struct.pack('!Q', 0))
                except Exception as e:
                    logger.warning(f"Error sending failure notification to {addr}: {str(e)}")
                return
            
            # Reset timeout for streaming
            conn.settimeout(None)
            
            # Try to open SHM to get shape/size info
            temp_shm = ImageStreamIO.open_shm(shm_name)
            if not temp_shm:
                logger.warning(f"SHM {shm_name} not available yet, waiting...")
                # Send error indication but keep monitoring
                try:
                    conn.send(struct.pack('!Q', 0))
                except Exception as e:
                    logger.warning(f"Error sending temp failure notification to {addr}: {str(e)}")
                    return
                
                # Wait for SHM to become available (up to 30 seconds)
                for _ in range(30):
                    time.sleep(1)
                    temp_shm = ImageStreamIO.open_shm(shm_name)
                    if temp_shm:
                        break
                else:
                    logger.error(f"SHM {shm_name} still not available after 30s wait, rejecting client {addr}")
                    return
            
            # Get frame size and shape
            frame_size = ImageStreamIO.get_data_size(temp_shm)
            if frame_size == 0:
                logger.error(f"Calculated frame size is 0 for SHM {shm_name}, rejecting client {addr}")
                try:
                    conn.send(struct.pack('!Q', 0))
                except Exception:
                    pass
                return
                
            # Send frame size
            conn.send(struct.pack('!Q', frame_size))
            
            # Send shape
            shape = ImageStreamIO.get_shape(temp_shm)
            if len(shape) == 2:
                conn.send(struct.pack('!II', *shape))
            else:
                logger.error(f"Invalid shape {shape} for SHM {shm_name}")
                return
                
            # Close temporary SHM
            if hasattr(temp_shm, 'close') and callable(temp_shm.close):
                temp_shm.close()
            
            # Bandwidth manager
            bw_manager = BandwidthManager(bw_limit_bps)
            
            # Protocol constants
            count_time_format = '!qd'  # int64, double
            count_time_size = struct.calcsize(count_time_format)
            
            # Main client loop
            running = True
            last_sent_frame = None
            
            while running and self.running:
                try:
                    # Check if buffer exists for this SHM
                    if shm_name not in self.buffers:
                        logger.warning(f"Buffer for SHM {shm_name} not found, waiting...")
                        time.sleep(0.5)
                        continue
                    
                    buffer = self.buffers[shm_name]
                    
                    # Wait for client command
                    cmd_data = conn.recv(4)
                    if not cmd_data or len(cmd_data) < 4:
                        logger.warning(f"Client {addr} disconnected or sent invalid command")
                        break
                    
                    cmd = struct.unpack('!I', cmd_data)[0]
                    
                    # Process command
                    if cmd == 1:  # Get latest frame
                        frames = buffer.get_newest_frames(1)
                        if not frames:
                            # No frames available yet
                            conn.send(struct.pack('!Q', 0))  # Send zero frame count
                            continue
                            
                        frame, count, timestamp = frames[0]
                        last_sent_frame = count
                        
                        # Send frame count
                        conn.send(struct.pack('!Q', 1))  # Sending 1 frame
                        
                        # Send frame
                        serialized_frame = frame.tobytes()
                        frame_data_len = len(serialized_frame)
                        
                        # Pack header for this frame
                        packed_count_time = struct.pack(count_time_format, count, timestamp)
                        total_payload_len = count_time_size + frame_data_len
                        header = struct.pack('!Q', total_payload_len)
                        
                        # Send header + count/time + frame
                        conn.sendall(header + packed_count_time + serialized_frame)
                        
                        # Update bandwidth manager
                        bytes_sent = 8 + total_payload_len
                        sleep_time = bw_manager.update(bytes_sent)
                        if sleep_time > 0:
                            time.sleep(sleep_time)
                    
                    elif cmd == 2:  # Get range of frames
                        # Receive range parameters
                        range_data = conn.recv(24)  # 3 int64 values
                        if not range_data or len(range_data) < 24:
                            logger.warning(f"Client {addr} sent incomplete range parameters")
                            continue
                            
                        start_frame, end_frame, max_frames = struct.unpack('!qqq', range_data)
                        
                        # Get frames from buffer
                        frames = buffer.get_frame_range(start_frame, end_frame, max_frames)
                        
                        # Send frame count
                        num_frames = len(frames)
                        conn.send(struct.pack('!Q', num_frames))
                        
                        # Send each frame
                        for frame, count, timestamp in frames:
                            last_sent_frame = count
                            
                            serialized_frame = frame.tobytes()
                            frame_data_len = len(serialized_frame)
                            
                            # Pack header for this frame
                            packed_count_time = struct.pack(count_time_format, count, timestamp)
                            total_payload_len = count_time_size + frame_data_len
                            header = struct.pack('!Q', total_payload_len)
                            
                            # Send header + count/time + frame
                            conn.sendall(header + packed_count_time + serialized_frame)
                            
                            # Update bandwidth manager
                            bytes_sent = 8 + total_payload_len
                            sleep_time = bw_manager.update(bytes_sent)
                            if sleep_time > 0:
                                time.sleep(sleep_time)
                    
                    elif cmd == 3:  # Get latest N frames
                        # Receive count parameter
                        count_data = conn.recv(8)  # 1 int64 value
                        if not count_data or len(count_data) < 8:
                            logger.warning(f"Client {addr} sent incomplete count parameter")
                            continue
                            
                        num_requested = struct.unpack('!q', count_data)[0]
                        
                        # Get frames from buffer
                        frames = buffer.get_newest_frames(num_requested)
                        
                        # Send frame count
                        num_frames = len(frames)
                        conn.send(struct.pack('!Q', num_frames))
                        
                        # Send each frame
                        for frame, count, timestamp in frames:
                            last_sent_frame = count
                            
                            serialized_frame = frame.tobytes()
                            frame_data_len = len(serialized_frame)
                            
                            # Pack header for this frame
                            packed_count_time = struct.pack(count_time_format, count, timestamp)
                            total_payload_len = count_time_size + frame_data_len
                            header = struct.pack('!Q', total_payload_len)
                            
                            # Send header + count/time + frame
                            conn.sendall(header + packed_count_time + serialized_frame)
                            
                            # Update bandwidth manager
                            bytes_sent = 8 + total_payload_len
                            sleep_time = bw_manager.update(bytes_sent)
                            if sleep_time > 0:
                                time.sleep(sleep_time)
                    
                    elif cmd == 4:  # Get buffer info
                        min_count, max_count = buffer.get_frame_counts()
                        buffer_size = buffer.get_size()
                        
                        # Pack buffer info: min_count, max_count, buffer_size
                        info_data = struct.pack('!qqq', 
                                              min_count if min_count is not None else -1,
                                              max_count if max_count is not None else -1, 
                                              buffer_size)
                        conn.send(info_data)
                    
                    else:
                        logger.warning(f"Client {addr} sent unknown command: {cmd}")
                    
                    # Reset bandwidth manager periodically
                    if time.time() - bw_manager.start_time > 1:
                        bw_manager.reset()
                
                except (ConnectionResetError, BrokenPipeError, ConnectionAbortedError) as e:
                    logger.warning(f"Client {addr} connection error: {str(e)}")
                    running = False
                    break
                except Exception as e:
                    logger.error(f"Error handling client {addr}: {str(e)}", exc_info=True)
                    running = False
                    break
                    
        except socket.timeout:
            logger.warning(f"Client {addr} timed out during handshake")
        except Exception as e:
            logger.error(f"Unexpected error with client {addr}: {str(e)}", exc_info=True)
        finally:
            try:
                conn.close()
            except Exception:
                pass
            logger.info(f"Handler for client {addr} terminated")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='SHM Buffering Server')
    parser.add_argument('--ip', type=str, default='127.0.0.1',
                        help='IP address to bind the server to (default: 127.0.0.1)')
    parser.add_argument('--port', type=int, default=5124,
                        help='Port to bind the server to (default: 5124)')
    parser.add_argument('--cores', type=str, default=None,
                        help='CPU core(s) to pin this process to (e.g., "0,1,2" or "3")')
    parser.add_argument('--bw-limit', type=float, default=10.0,
                        help='Maximum bandwidth per client in MB/s (default: 10.0)')
    parser.add_argument('--max-clients', type=int, default=None,
                        help='Maximum number of concurrent clients (default: unlimited)')
    parser.add_argument('--buffer-size', type=int, default=100,
                        help='Maximum number of frames to buffer per SHM (default: 100)')
    parser.add_argument('--poll-interval', type=float, default=0.01,
                        help='Interval between SHM polls in seconds. Set to 0 or less to poll as fast as possible (high performance mode). (default: 0.01)')
    parser.add_argument('--log-level', type=str, default='INFO',
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                        help='Set the logging level (default: INFO)')

    args = parser.parse_args()

    log_level = getattr(logging, args.log_level.upper(), logging.INFO)
    coloredlogs.install(
        level=log_level,
        fmt='%(asctime)s [%(levelname)s] %(name)s:%(funcName)s - %(message)s',
        level_styles={'debug': {'color': 'green'}, 'info': {'color': 'cyan'}, 'warning': {'color': 'yellow'}, 'error': {'color': 'red'}, 'critical': {'bold': True, 'color': 'red'}}
    )
    logger = logging.getLogger("ShmBufferingServer")

    # Add a fallback StreamHandler to ensure logs are printed to the console
    if not logger.hasHandlers():
        console_handler = logging.StreamHandler()
        console_handler.setLevel(log_level)
        console_formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(name)s:%(funcName)s - %(message)s')
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)
        logger.info("Fallback StreamHandler added to logger.")

    if args.cores:
        try:
            cores = [int(core.strip()) for core in args.cores.split(',')]
            pin_to_cores(cores)
        except ValueError:
            logger.error(f"Invalid core specification: {args.cores}. Format should be comma-separated integers.")
        except Exception as e:
            logger.error(f"Error setting up CPU pinning: {e}")
    
    server = ShmBufferingServer(
        host=args.ip, 
        port=args.port, 
        max_clients=args.max_clients, 
        bw_limit_mbps=args.bw_limit,
        buffer_size=args.buffer_size,
        poll_interval=args.poll_interval,
        logger=logger
    )
    server.start()