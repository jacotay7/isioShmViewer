#!/usr/bin/env python3
import socket
import struct
import numpy as np
import time
import threading
import argparse
import os
import logging
import coloredlogs
from datetime import datetime
import queue
import signal
import sys

# Add parent directory to path to import SHMStream
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from shm_stream import SHMStream

# Setup logger
logger = logging.getLogger(__name__)

class FrameCapturer:
    """
    Client that captures frames from a SHM server and saves them to a binary file.
    """
    def __init__(self, shm_name, server_ip='127.0.0.1', server_port=5124, 
                 target_fps=30, max_size_mb=1000, output_dir=None):
        self.shm_name = shm_name
        self.server_ip = server_ip
        self.server_port = server_port
        self.target_fps = target_fps
        self.target_interval = 1.0 / target_fps
        self.max_size_bytes = max_size_mb * 1024 * 1024  # Convert MB to bytes
        self.output_dir = output_dir or os.getcwd()
        
        # Ensure output directory exists
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Generate output filename based on current timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_file = os.path.join(
            self.output_dir, f"{shm_name}_{timestamp}.bin"
        )
        
        # Queue for frame data
        self.frame_queue = queue.Queue(maxsize=100)  # Buffer up to 100 frames
        
        # Stats
        self.total_frames = 0
        self.total_bytes = 0
        self.start_time = None
        self.running = False
        self.stats_lock = threading.Lock()
        self.frame_shape = None
        self.frame_size = None
        self.late_frames = 0
        
        # SHMStream object
        self.stream = None
        
        # File header format (to be saved at the beginning of the binary file)
        # This will help us read the file back later
        self.file_header_format = {
            'magic': b'SHMCAPT',           # 7 bytes
            'version': 1,                  # 1 byte (uint8)
            'shm_name_len': 0,             # 1 byte (uint8)
            'shm_name': b'',               # Variable
            'timestamp': 0,                # 8 bytes (int64)
            'target_fps': target_fps,      # 4 bytes (float32)
            'height': 0,                   # 4 bytes (uint32)
            'width': 0,                    # 4 bytes (uint32)
            'dtype_code': 0,               # 1 byte (uint8)
            'dtype_size': 0,               # 1 byte (uint8)
        }
        
        # Map numpy dtypes to codes for file header
        self.dtype_codes = {
            np.dtype('float32'): 1,
            np.dtype('float64'): 2,
            np.dtype('uint8'): 3,
            np.dtype('uint16'): 4,
            np.dtype('int16'): 5,
            np.dtype('int32'): 6,
            np.dtype('uint32'): 7,
        }
        
        # Add initialization of thread attributes
        self.capture_thread = None
        self.writer_thread = None
        self.stats_thread = None
    
    def connect(self):
        """
        Connect to the SHM server using SHMStream.
        """
        try:
            # Create SHMStream object to handle connection
            self.stream = SHMStream(
                self.shm_name,
                server_ip=self.server_ip,
                port=self.server_port,
                server_type=SHMStream.SERVER_TYPE_BUFFER if self.server_port == 5124 else SHMStream.SERVER_TYPE_VIEWER,
                buffer_size=10,  # Keep minimal buffer since we're saving frames immediately
                poll_interval=self.target_interval  # Set poll interval to match target FPS
            )
            
            # Register error callback
            self.stream.register_error_callback(self._on_connection_error)
            
            # Connect to the server
            success = self.stream.connect()
            
            # Wait for connection to be established
            timeout = 10  # 10 seconds timeout
            start_time = time.time()
            while time.time() - start_time < timeout:
                if self.stream.connected:
                    # Retrieve the frame shape
                    self.frame_shape = self.stream.image_shape
                    if self.frame_shape:
                        height, width = self.frame_shape
                        # Estimate frame size (assuming float32 data)
                        self.frame_size = height * width * 4  # 4 bytes per float32
                        
                        logger.info(f"Connected to {self.server_ip}:{self.server_port}")
                        logger.info(f"SHM: {self.shm_name}, Shape: {self.frame_shape}, Size: {self.frame_size} bytes")
                        
                        # If using viewer server, only register callback with throttling
                        if self.stream.server_type == SHMStream.SERVER_TYPE_VIEWER:
                            logger.info(f"Using frame callback with rate limiting at {self.target_fps} FPS")
                            # We'll handle throttling in our own callback instead of automatic callbacks
                            self._last_frame_time = 0
                            self.stream.register_frame_callback(self._throttled_frame_callback)
                            
                        return True
                elif self.stream.connection_error:
                    logger.error(f"Connection error: {self.stream.connection_error}")
                    return False
                
                time.sleep(0.1)
            
            logger.error(f"Timeout waiting for connection to {self.server_ip}:{self.server_port}")
            return False
            
        except Exception as e:
            logger.error(f"Failed to connect to server: {str(e)}")
            return False
    
    def _throttled_frame_callback(self, frame, frame_number, timestamp):
        """
        Callback that throttles incoming frames to the target frame rate
        """
        current_time = time.time()
        
        # Only process frames at the target frame rate
        if current_time - self._last_frame_time >= self.target_interval:
            self._last_frame_time = current_time
            self._on_frame_received(frame, frame_number, timestamp)
    
    def _on_connection_error(self, error_msg):
        """
        Callback for SHMStream connection errors
        """
        logger.error(f"Stream connection error: {error_msg}")
        self.running = False
    
    def _on_frame_received(self, frame, frame_number, timestamp):
        """
        Callback for frames received from SHMStream
        """
        try:
            if not self.running:
                return
                
            frame_data = {
                'frame': frame,
                'frame_number': frame_number,
                'timestamp': timestamp,
                'receive_time': time.time()
            }
            
            # Put the frame in the queue, but don't block if full
            try:
                self.frame_queue.put_nowait(frame_data)
                logger.debug(f"Received frame {frame_number} with shape {frame.shape}")
            except queue.Full:
                logger.warning("Frame queue is full, dropping frame")
                
        except Exception as e:
            logger.error(f"Error in frame callback: {str(e)}")
    
    def request_frame(self):
        """
        Request a single frame from the server using SHMStream.
        This is only used for buffer server mode or manual frame requests.
        """
        try:
            if self.stream.server_type == SHMStream.SERVER_TYPE_BUFFER:
                # Request latest frame from buffer server
                self.stream.request_latest_frame()
            
            # Get the latest frame
            frame, frame_number, timestamp = self.stream.get_frame()
            
            if frame is None:
                logger.warning("No frame available")
                return None
            
            return {
                'frame': frame,
                'frame_number': frame_number,
                'timestamp': timestamp,
                'receive_time': time.time()
            }
            
        except Exception as e:
            logger.error(f"Error receiving frame: {str(e)}")
            return None
    
    def _capture_thread(self):
        """
        Thread that captures frames at the target framerate.
        """
        logger.info(f"Starting capture thread with target {self.target_fps} FPS")
        next_frame_time = time.time()
        
        while self.running:
            current_time = time.time()
            
            # Check if we need to get a frame now
            if current_time >= next_frame_time:
                # For buffer server, request a frame
                # For viewer server, we're using the throttled callback
                if self.stream.server_type == SHMStream.SERVER_TYPE_BUFFER:
                    frame_data = self.request_frame()
                    
                    if frame_data:
                        try:
                            self.frame_queue.put_nowait(frame_data)
                        except queue.Full:
                            logger.warning("Frame queue is full, dropping frame")
                
                # Schedule next frame time
                next_frame_time += self.target_interval
                
                # If we've fallen behind schedule, reset next frame time
                if next_frame_time < current_time:
                    skipped = int((current_time - next_frame_time) / self.target_interval) + 1
                    with self.stats_lock:
                        self.late_frames += skipped
                    logger.warning(f"Capture is {skipped} frames behind schedule. Resetting timing.")
                    next_frame_time = current_time + self.target_interval
            
            # Calculate sleep time to maintain frame rate
            sleep_time = next_frame_time - time.time()
            if sleep_time > 0:
                time.sleep(min(sleep_time, 0.01))  # Sleep at most 10ms at a time
            else:
                # Small sleep to avoid tight loop
                time.sleep(0.001)
    
    def _writer_thread(self):
        """
        Thread that writes frames to the binary file.
        """
        logger.info(f"Starting writer thread, saving to {self.output_file}")
        
        try:
            # Wait for first frame with a longer timeout since we're just starting
            first_frame_wait_time = 30  # 30 seconds to get the first frame
            start_wait = time.time()
            
            logger.info(f"Waiting up to {first_frame_wait_time} seconds for first frame...")
            
            # Keep trying until we get a frame or timeout
            while time.time() - start_wait < first_frame_wait_time:
                try:
                    # Check if we have a frame in the queue with a short timeout
                    first_frame_data = self.frame_queue.get(timeout=1.0)
                    break
                except queue.Empty:
                    if not self.running:
                        logger.warning("Capture stopped while waiting for first frame")
                        return
                    # Request a frame explicitly
                    if self.stream.server_type == SHMStream.SERVER_TYPE_BUFFER:
                        logger.debug("No frame yet, requesting buffer info...")
                        self.stream.request_buffer_info()
                        if self.stream.server_buffer_size > 0:
                            logger.debug(f"Server has {self.stream.server_buffer_size} frames, requesting latest...")
                            self.stream.request_latest_frame()
                    logger.debug("Waiting for first frame...")
                    continue
            else:
                # If we exit the while loop without breaking, we've timed out
                logger.error(f"Timed out waiting for first frame after {first_frame_wait_time} seconds")
                self.running = False
                return
            
            # Now we have the first frame, open the file and write the header
            with open(self.output_file, 'wb') as f:
                first_frame = first_frame_data['frame']
                
                # Update file header with frame info
                self.file_header_format['timestamp'] = int(time.time())
                self.file_header_format['shm_name_len'] = len(self.shm_name)
                self.file_header_format['shm_name'] = self.shm_name.encode()
                self.file_header_format['height'], self.file_header_format['width'] = first_frame.shape
                
                if first_frame.dtype in self.dtype_codes:
                    self.file_header_format['dtype_code'] = self.dtype_codes[first_frame.dtype]
                else:
                    logger.warning(f"Unknown dtype {first_frame.dtype}, using code 0")
                    
                self.file_header_format['dtype_size'] = first_frame.dtype.itemsize
                
                # Write file header
                f.write(self.file_header_format['magic'])  # 7 bytes
                f.write(struct.pack('B', self.file_header_format['version']))  # 1 byte
                f.write(struct.pack('B', self.file_header_format['shm_name_len']))  # 1 byte
                f.write(self.file_header_format['shm_name'])  # Variable
                f.write(struct.pack('!q', self.file_header_format['timestamp']))  # 8 bytes
                f.write(struct.pack('!f', self.file_header_format['target_fps']))  # 4 bytes
                f.write(struct.pack('!II', self.file_header_format['height'], self.file_header_format['width']))  # 8 bytes
                f.write(struct.pack('BB', self.file_header_format['dtype_code'], self.file_header_format['dtype_size']))  # 2 bytes
                
                # Write first frame
                self._write_frame(f, first_frame_data)
                
                # Process remaining frames
                while self.running:
                    try:
                        frame_data = self.frame_queue.get(timeout=1.0)
                        self._write_frame(f, frame_data)
                        
                        # Check if we've exceeded max file size
                        if self.total_bytes >= self.max_size_bytes:
                            logger.info(f"Reached maximum file size of {self.max_size_mb} MB. Stopping capture.")
                            self.running = False
                            break
                            
                    except queue.Empty:
                        if not self.running:
                            break
                        continue
                        
                logger.info(f"Writer thread finished. Total frames: {self.total_frames}, Total size: {self.total_bytes / (1024*1024):.2f} MB")
                
        except Exception as e:
            logger.error(f"Writer thread error: {str(e)}", exc_info=True)
            self.running = False
    
    def _write_frame(self, file, frame_data):
        """
        Write a single frame and its metadata to the file.
        """
        frame = frame_data['frame']
        frame_number = frame_data['frame_number']
        timestamp = frame_data['timestamp']
        
        # Write frame metadata (frame number, timestamp)
        file.write(struct.pack('!qd', frame_number, timestamp))
        
        # Write frame data
        frame_bytes = frame.tobytes()
        file.write(frame_bytes)
        
        # Update stats
        with self.stats_lock:
            self.total_frames += 1
            bytes_written = 16 + len(frame_bytes)  # 16 bytes for metadata + frame data
            self.total_bytes += bytes_written
            
            # Periodic stats reporting
            if self.total_frames % 30 == 0:  # Report every 30 frames (approx 1 second at 30fps)
                elapsed = time.time() - self.start_time
                actual_fps = self.total_frames / elapsed if elapsed > 0 else 0
                mb_written = self.total_bytes / (1024 * 1024)
                mb_per_sec = mb_written / elapsed if elapsed > 0 else 0
                
                logger.info(f"Stats: {self.total_frames} frames, {mb_written:.2f} MB written, "
                           f"Actual FPS: {actual_fps:.2f}, Data rate: {mb_per_sec:.2f} MB/s, "
                           f"Late frames: {self.late_frames}")
    
    def _stats_thread(self):
        """
        Thread that periodically reports statistics.
        """
        last_frames = 0
        last_bytes = 0
        last_time = time.time()
        
        while self.running:
            time.sleep(5.0)  # Report every 5 seconds
            
            current_time = time.time()
            with self.stats_lock:
                elapsed = current_time - last_time
                frames = self.total_frames - last_frames
                bytes_written = self.total_bytes - last_bytes
                
                if elapsed > 0:
                    fps = frames / elapsed
                    mbps = (bytes_written / (1024 * 1024)) / elapsed
                    queue_size = self.frame_queue.qsize()
                    file_size_mb = self.total_bytes / (1024 * 1024)
                    percent_full = (self.total_bytes / self.max_size_bytes) * 100
                    
                    logger.info(f"Last {elapsed:.1f}s: {fps:.2f} FPS, {mbps:.2f} MB/s, "
                               f"Queue: {queue_size}, Total: {file_size_mb:.2f} MB ({percent_full:.1f}% of max), "
                               f"Late: {self.late_frames}")
                
                last_frames = self.total_frames
                last_bytes = self.total_bytes
                last_time = current_time
    
    def start(self):
        """
        Start capturing frames.
        """
        if not self.connect():
            return False
        
        self.running = True
        self.start_time = time.time()
        
        # Start capture thread for Buffer servers
        # For Viewer servers, we're using the throttled callback
        if self.stream.server_type == SHMStream.SERVER_TYPE_BUFFER:
            self.capture_thread = threading.Thread(target=self._capture_thread)
            self.capture_thread.daemon = True
            self.capture_thread.start()
        
        # Start writer thread
        self.writer_thread = threading.Thread(target=self._writer_thread)
        self.writer_thread.daemon = True
        self.writer_thread.start()
        
        # Start stats thread
        self.stats_thread = threading.Thread(target=self._stats_thread)
        self.stats_thread.daemon = True
        self.stats_thread.start()
        
        return True
    
    def stop(self):
        """
        Stop capturing frames and close the connection.
        """
        if not self.running:  # Avoid stopping multiple times
            return
            
        logger.info("Stopping capture...")
        self.running = False
        
        if hasattr(self, 'capture_thread') and self.capture_thread:
            self.capture_thread.join(timeout=2.0)
        
        if hasattr(self, 'writer_thread') and self.writer_thread:
            self.writer_thread.join(timeout=5.0)
            
        if hasattr(self, 'stats_thread') and self.stats_thread:
            self.stats_thread.join(timeout=1.0)
        
        if self.stream:
            self.stream.close()
            self.stream = None
        
        # Final stats
        elapsed = time.time() - self.start_time if self.start_time else 0
        if elapsed > 0:
            logger.info(f"Capture finished. Total: {self.total_frames} frames in {elapsed:.2f}s "
                       f"({self.total_frames/elapsed:.2f} FPS), "
                       f"File size: {self.total_bytes/(1024*1024):.2f} MB")
            
        # Print a nice summary about the output file
        if self.total_frames > 0 and os.path.exists(self.output_file):
            file_size_mb = os.path.getsize(self.output_file) / (1024 * 1024)
            logger.info(f"Output file created: {self.output_file}")
            logger.info(f"File summary: {file_size_mb:.2f} MB, {self.total_frames} frames, "
                       f"SHM: {self.shm_name}, Shape: {self.frame_shape}")


def signal_handler(sig, frame):
    """Handle interrupt signals."""
    logger.info("Received interrupt signal, shutting down...")
    if 'capturer' in globals():
        try:
            capturer.stop()
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")
    sys.exit(0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='SHM Frame Capture Client')
    parser.add_argument('shm_name', type=str, help='Name of the shared memory to capture')
    parser.add_argument('--ip', type=str, default='127.0.0.1', help='Server IP address (default: 127.0.0.1)')
    parser.add_argument('--port', type=int, default=5123, help='Server port (default: 5123)')
    parser.add_argument('--fps', type=float, default=30.0, help='Target frames per second (default: 30.0)')
    parser.add_argument('--max-size', type=float, default=1000.0, help='Maximum output file size in MB (default: 1000.0)')
    parser.add_argument('--output-dir', type=str, default=None, help='Output directory for captured frames (default: current directory)')
    parser.add_argument('--log-level', type=str, default='INFO',
                      choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                      help='Set the logging level (default: INFO)')
    
    args = parser.parse_args()
    
    # Configure logger
    log_level = getattr(logging, args.log_level.upper(), logging.INFO)
    coloredlogs.install(
        level=log_level,
        fmt='%(asctime)s [%(levelname)s] %(name)s:%(funcName)s - %(message)s',
        level_styles={'debug': {'color': 'green'}, 'info': {'color': 'cyan'}, 
                      'warning': {'color': 'yellow'}, 'error': {'color': 'red'}, 
                      'critical': {'bold': True, 'color': 'red'}}
    )
    
    # Add a fallback StreamHandler to ensure logs are printed to the console
    if not logger.hasHandlers():
        console_handler = logging.StreamHandler()
        console_handler.setLevel(log_level)
        console_formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(name)s:%(funcName)s - %(message)s')
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)
        logger.info("Fallback StreamHandler added to logger.")
    
    # Register signal handler for clean shutdown
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Create and start the capturer
    capturer = FrameCapturer(
        shm_name=args.shm_name,
        server_ip=args.ip,
        server_port=args.port,
        target_fps=args.fps,
        max_size_mb=args.max_size,
        output_dir=args.output_dir
    )
    
    if capturer.start():
        logger.info(f"Capturing frames from {args.shm_name} at {args.fps} FPS")
        try:
            # Keep main thread running until stopped
            while capturer.running:
                time.sleep(0.5)
        except KeyboardInterrupt:
            logger.info("Keyboard interrupt received, shutting down...")
            capturer.stop()
        except Exception as e:
            logger.error(f"Error in main loop: {e}")
            capturer.stop()
    else:
        logger.error("Failed to start capture")
        sys.exit(1)
    
    logger.info("Capture client terminated")


#%%

# %%
