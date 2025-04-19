#!/usr/bin/env python3
import socket
import struct
import threading
import numpy as np
import time

class SHMStream:
    """
    A universal SHM stream class that can connect to both viewer and buffering servers.
    
    This class handles the network communication, data formatting, and optionally maintains
    a local buffer of received frames. It can automatically build up a contiguous data buffer
    using frame counts, handling missing frames appropriately.
    """
    
    # Server types
    SERVER_TYPE_VIEWER = 1
    SERVER_TYPE_BUFFER = 2
    
    def __init__(self, shm_name, server_ip="127.0.0.1", port=5123, 
                 server_type=None, custom_shape=None, buffer_size=100,
                 poll_interval=0.1):
        """
        Initialize a connection to a SHM stream server.
        
        Args:
            shm_name (str): Name of the shared memory to connect to
            server_ip (str): Server IP address
            port (int): Server port (default 5123 for viewer, 5124 for buffer)
            server_type (int): SERVER_TYPE_VIEWER or SERVER_TYPE_BUFFER, or None to auto-detect
            custom_shape (tuple): Custom shape to reshape frames to, or None to use server shape
            buffer_size (int): Size of the local buffer to maintain (number of frames)
            poll_interval (float): Interval in seconds for polling buffer server (default 0.1)
        """
        self.shm_name = shm_name
        self.server_ip = server_ip
        self.port = port
        self.custom_shape = custom_shape
        self.max_buffer_size = buffer_size
        self.poll_interval = poll_interval
        
        # Auto-detect server type based on port if not specified
        if server_type is None:
            self.server_type = (self.SERVER_TYPE_VIEWER if port == 5123 
                               else self.SERVER_TYPE_BUFFER)
        else:
            self.server_type = server_type
        
        # Connection state
        self.socket = None
        self.socket_lock = threading.Lock()
        self.connected = False
        self.connection_error = None
        self.image_shape = None
        
        # Data buffer
        self.frames_buffer = {}  # Dictionary mapping frame_count to (frame, timestamp)
        self.frame_counts = []   # Ordered list of frame counts in buffer
        
        # Buffer metadata
        self.buffer_min_frame = -1
        self.buffer_max_frame = -1
        self.server_buffer_size = 0
        
        # Latest frame info
        self.latest_frame = None
        self.latest_count = -1
        self.latest_time = -1.0
        
        # Define struct formats for count and time
        self.count_time_format = '!qd'  # int64, float64
        self.count_time_size = struct.calcsize(self.count_time_format)
        
        # Callbacks
        self.on_frame_callback = None
        self.on_buffer_info_callback = None
        self.on_connection_error_callback = None
        
    def connect(self):
        """
        Connect to the server and initialize the stream. 
        Returns True if successful, False otherwise.
        """
        threading.Thread(target=self._connect_thread, daemon=True).start()
        return True
        
    def _connect_thread(self):
        """Thread function for connecting to the server"""
        try:
            # Create socket
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.settimeout(10)  # 10 second timeout for initial connection
            
            # Connect
            self.socket.connect((self.server_ip, self.port))
            
            # Send SHM name (padded to 256 bytes)
            name_bytes = self.shm_name.encode()
            name_padded = name_bytes.ljust(256, b'\x00')
            self.socket.sendall(name_padded)
            
            # Receive initial frame size (8 bytes)
            data = self.recv_all(self.socket, 8)
            if data is None:
                raise ConnectionError("Failed to receive frame size from server")
                
            expected_frame_size = struct.unpack('!Q', data)[0]
            if expected_frame_size == 0:
                raise ConnectionError(f"Invalid SHM name: {self.shm_name}. No shared memory found.")
                
            # Receive the image shape (e.g., 2 integers for 2D shape)
            shape_data = self.recv_all(self.socket, 8)  
            if shape_data is None:
                raise ConnectionError("Failed to receive image shape from server")
                
            self.image_shape = struct.unpack('!II', shape_data)
            
            # Reset timeout for normal operation
            self.socket.settimeout(None)
            
            # Mark as connected
            self.connected = True
            
            if self.server_type == self.SERVER_TYPE_VIEWER:
                # For viewer server, start a background thread to continuously receive frames
                threading.Thread(target=self._viewer_receive_thread, daemon=True).start()
            elif self.server_type == self.SERVER_TYPE_BUFFER:
                # For buffer server, start threads for info updates and continuous polling
                threading.Thread(target=self._buffer_info_thread, daemon=True).start()
                threading.Thread(target=self._buffer_poll_thread, daemon=True).start()
            
        except socket.timeout:
            self._handle_connection_error(f"Connection timed out: {self.server_ip}:{self.port}")
        except ConnectionRefusedError:
            self._handle_connection_error(f"Connection refused: {self.server_ip}:{self.port}")
        except ConnectionError as e:
            self._handle_connection_error(str(e))
        except Exception as e:
            self._handle_connection_error(f"Connection error: {str(e)}")
            
    def _handle_connection_error(self, error_msg):
        """Handle connection errors"""
        self.connected = False
        self.connection_error = error_msg
        if self.on_connection_error_callback:
            self.on_connection_error_callback(error_msg)
            
    def _viewer_receive_thread(self):
        """Thread to continuously receive frames from a viewer server"""
        try:
            while self.connected:
                # Receive header: 8 bytes indicating the total length of the payload
                header = self.recv_all(self.socket, 8)
                if header is None:
                    raise ConnectionError("Connection closed by server")
                    
                total_payload_length = struct.unpack('!Q', header)[0]
                
                if total_payload_length < self.count_time_size:
                    print(f"Warning: Payload too small: {total_payload_length} bytes")
                    continue
                
                # Receive the payload
                payload_bytes = self.recv_all(self.socket, total_payload_length)
                if payload_bytes is None:
                    raise ConnectionError("Failed to receive complete payload")
                    
                # Process the frame
                self._process_frame_data(payload_bytes)
                
        except Exception as e:
            self._handle_connection_error(f"Error receiving data: {str(e)}")
    
    def _buffer_info_thread(self):
        """Thread for periodically updating buffer information from a buffer server"""
        while self.connected:
            try:
                self.request_buffer_info()
                time.sleep(1)  # Update every second
            except Exception as e:
                self._handle_connection_error(f"Buffer info error: {str(e)}")
                break

    def _buffer_poll_thread(self):
        """Thread for continuously polling frames from a buffer server to fill the local buffer."""
        while self.connected:
            try:
                # Determine the next frame count needed by the client buffer
                # If buffer is empty (buffer_max_frame is -1), start requesting from frame 0
                next_frame_needed = self.buffer_max_frame + 1 if self.frame_counts else 0
                
                # Define the end frame for the request (effectively infinity for int64)
                # The server will only send up to max_frames_per_request anyway
                end_frame_request = (1 << 63) - 1 
                
                # Define how many frames to request at most in one go.
                # Using max_buffer_size might be large, but server limits response.
                # Alternatively, a smaller fixed chunk size (e.g., 100 or server_buffer_size) could be used.
                # Let's use max_buffer_size for now.
                max_frames_per_request = self.max_buffer_size 
                
                # Request the range of frames starting from the next needed one
                success = self.request_frame_range(
                    next_frame_needed, 
                    end_frame_request, 
                    max_frames_per_request
                )
                # Optional: Add logging or checks based on success status if needed
                
                # Wait for the specified interval before polling again
                # If the request returned many frames, might want a shorter wait?
                # If it returned few/none, might want a longer wait?
                # Simple fixed interval for now.
                time.sleep(self.poll_interval)
                
            except Exception as e:
                # Handle potential errors during polling, e.g., connection issues
                self._handle_connection_error(f"Buffer polling error: {str(e)}")
                break # Exit thread on error
            
            # Add a small sleep even if polling interval is very short to prevent busy-waiting
            if self.poll_interval <= 0:
                time.sleep(0.01) 
    
    def recv_all(self, sock, n):
        """Helper function to receive exactly n bytes from a socket."""
        data = b''
        while len(data) < n:
            packet = sock.recv(n - len(data))
            if not packet:
                return None
            data += packet
        return data
    
    def _process_frame_data(self, payload_bytes):
        """
        Process received frame data.
        
        Args:
            payload_bytes: The raw bytes containing count, timestamp, and frame data
        """
        try:
            # Extract count and time from the beginning of payload
            frame_count, frame_time = struct.unpack(
                self.count_time_format, 
                payload_bytes[:self.count_time_size]
            )
            
            # The rest is the frame data
            frame_bytes = payload_bytes[self.count_time_size:]
            
            # Convert bytes to numpy array
            frame = np.frombuffer(frame_bytes, dtype=np.float32)
            
            # Reshape using the shape we received at connection time
            target_shape = self.custom_shape if self.custom_shape else self.image_shape
            if not target_shape:
                raise ValueError("Target shape unknown")
                
            # Check buffer size against target shape
            expected_elements = np.prod(target_shape)
            if frame.size != expected_elements:
                raise ValueError(f"Frame buffer size mismatch. Expected {expected_elements} elements, got {frame.size}")
                
            frame = frame.reshape(target_shape)
            
            # Update latest frame info
            self.latest_frame = frame
            self.latest_count = frame_count
            self.latest_time = frame_time
            
            # Add to buffer and maintain buffer size
            self._add_to_buffer(frame, frame_count, frame_time)
            
            # Call the callback if set
            if self.on_frame_callback:
                self.on_frame_callback(frame, frame_count, frame_time)
                
        except Exception as e:
            print(f"Error processing frame: {e}")
    
    def _add_to_buffer(self, frame, frame_count, frame_time):
        """
        Add a frame to the buffer and maintain buffer size limit strictly.
        
        Args:
            frame: The numpy array with frame data
            frame_count: The frame counter value
            frame_time: The timestamp of the frame
        """
        # If frame already exists, update it (optional, could skip if performance critical)
        if frame_count in self.frames_buffer:
            self.frames_buffer[frame_count] = (frame, frame_time)
            # No need to adjust frame_counts or size limit if just updating
            return

        # Ensure buffer does not exceed max size BEFORE adding
        # Only remove if max_buffer_size is positive
        while self.max_buffer_size > 0 and len(self.frames_buffer) >= self.max_buffer_size:
            # Remove oldest frame if buffer is full
            if self.frame_counts:
                oldest_count = self.frame_counts.pop(0) # Remove from sorted list first
                if oldest_count in self.frames_buffer:
                    del self.frames_buffer[oldest_count] # Then remove from dictionary
            else:
                # Should not happen if len(self.frames_buffer) >= max_buffer_size > 0
                # but break just in case to avoid infinite loop
                break 
        
        # Add the new frame if buffer size allows (or if size is unlimited)
        if self.max_buffer_size <= 0 or len(self.frames_buffer) < self.max_buffer_size:
            # Add to buffer dictionary
            self.frames_buffer[frame_count] = (frame, frame_time)
            
            # Update ordered frame count list by inserting in sorted order
            pos = self._find_insertion_point(self.frame_counts, frame_count)
            self.frame_counts.insert(pos, frame_count)
        
        # Update min/max frame counts based on the current state of frame_counts
        if self.frame_counts:
            self.buffer_min_frame = self.frame_counts[0]
            self.buffer_max_frame = self.frame_counts[-1]
        else:
            self.buffer_min_frame = -1
            self.buffer_max_frame = -1
    
    def _find_insertion_point(self, sorted_list, value):
        """
        Find the insertion point for value in a sorted list.
        
        Args:
            sorted_list: A sorted list
            value: The value to insert
            
        Returns:
            The index where value should be inserted
        """
        left, right = 0, len(sorted_list)
        while left < right:
            mid = (left + right) // 2
            if sorted_list[mid] < value:
                left = mid + 1
            else:
                right = mid
        return left
    
    def get_contiguous_buffer(self, start_frame=None, end_frame=None, max_frames=None):
        """
        Create a contiguous buffer array potentially filling missing frames with NaN.
        The default range spans at most max_buffer_size frames ending at the latest frame.
        
        Args:
            start_frame: Starting frame number (default: calculated based on max_buffer_size)
            end_frame: Ending frame number (default: buffer_max_frame)
            max_frames: Maximum number of frames to include (overrides default range size)
            
        Returns:
            Tuple of (frames_array, counts_array, times_array) where:
                frames_array: A 3D numpy array of shape (num_frames, height, width)
                counts_array: A 1D array of frame counts
                times_array: A 1D array of frame timestamps
        """
        if not self.frames_buffer or not self.frame_counts: # Check frame_counts too
            return None, None, None
            
        # Determine default end_frame
        effective_end_frame = end_frame if end_frame is not None else self.buffer_max_frame
        
        # Determine default start_frame, ensuring the default range <= max_buffer_size
        if start_frame is None:
            # Calculate start based on end and max buffer size, but not before the actual oldest frame
            calculated_start = effective_end_frame - self.max_buffer_size + 1
            effective_start_frame = max(self.buffer_min_frame, calculated_start)
        else:
            effective_start_frame = start_frame

        # Ensure start is not after end
        if effective_start_frame > effective_end_frame:
             effective_start_frame = effective_end_frame # Or return empty? Return single frame for now.

        # Apply max_frames constraint if specified - this limits the user's request further
        if max_frames is not None:
             requested_num_frames = effective_end_frame - effective_start_frame + 1
             if requested_num_frames > max_frames:
                 # Adjust start_frame forward if max_frames is smaller than the default/specified range
                 effective_start_frame = effective_end_frame - max_frames + 1
                 # Re-ensure start frame is not before the absolute minimum available
                 effective_start_frame = max(self.buffer_min_frame, effective_start_frame)

        # Calculate the final number of frames based on effective start/end
        num_frames = effective_end_frame - effective_start_frame + 1
        if num_frames <= 0:
            return None, None, None
            
        # Get one frame to determine shape and dtype
        # Use the latest frame for potentially better dtype/shape accuracy
        sample_frame, _ = self.frames_buffer[self.buffer_max_frame]
        frame_shape = sample_frame.shape
        frame_dtype = sample_frame.dtype # Use the actual dtype
        
        # Create output arrays
        frames_array = np.full((num_frames,) + frame_shape, np.nan, dtype=frame_dtype) # Use actual dtype
        counts_array = np.arange(effective_start_frame, effective_end_frame + 1, dtype=np.int64) # Use int64 for counts
        times_array = np.full(num_frames, np.nan, dtype=np.float64)
        
        # Fill arrays with available data
        for i, count in enumerate(counts_array):
            if count in self.frames_buffer:
                frame, timestamp = self.frames_buffer[count]
                # Ensure frame shape matches before assignment (optional safety check)
                if frame.shape == frame_shape and frame.dtype == frame_dtype:
                     frames_array[i] = frame
                     times_array[i] = timestamp
                else:
                     print(f"Warning: Frame {count} shape/dtype mismatch. Expected {frame_shape}/{frame_dtype}, got {frame.shape}/{frame.dtype}. Skipping.")

        return frames_array, counts_array, times_array
    
    def get_frame(self, frame_count=None):
        """
        Get a specific frame from the buffer.
        
        Args:
            frame_count: The frame count to retrieve, or None for the latest frame
            
        Returns:
            Tuple of (frame, count, timestamp) or (None, None, None) if not found
        """
        if frame_count is None:
            return (self.latest_frame, self.latest_count, self.latest_time)
            
        if frame_count in self.frames_buffer:
            frame, timestamp = self.frames_buffer[frame_count]
            return (frame, frame_count, timestamp)
            
        return (None, None, None)
    
    # --- Buffer server specific methods ---
    
    def request_latest_frame(self):
        """Request the latest frame from a buffer server"""
        if not self.connected or self.server_type != self.SERVER_TYPE_BUFFER:
            return False
            
        try:
            with self.socket_lock:
                # Send command 1 (Get latest frame)
                self.socket.sendall(struct.pack('!I', 1))
                
                # Receive frame count
                frame_count_data = self.recv_all(self.socket, 8)
                if not frame_count_data:
                    raise ConnectionError("Failed to receive frame count")
                    
                frame_count = struct.unpack('!Q', frame_count_data)[0]
                if frame_count == 0:
                    # No frames available
                    return False
                
                # We know we're getting exactly one frame
                self._receive_buffer_frame()
                return True
                    
        except Exception as e:
            self._handle_connection_error(f"Error requesting latest frame: {str(e)}")
            return False
            
    def request_frame_range(self, start_frame, end_frame, max_frames=None):
        """
        Request a range of frames from a buffer server.
        
        Args:
            start_frame: First frame number to request
            end_frame: Last frame number to request
            max_frames: Maximum number of frames to request
        
        Returns:
            Boolean indicating success
        """
        if not self.connected or self.server_type != self.SERVER_TYPE_BUFFER:
            return False
            
        if max_frames is None:
            max_frames = end_frame - start_frame + 1
            
        try:
            with self.socket_lock:
                # Send command 2 (Get frame range)
                self.socket.sendall(struct.pack('!I', 2))
                
                # Send range parameters
                self.socket.sendall(struct.pack('!qqq', start_frame, end_frame, max_frames))
                
                # Receive frame count
                frame_count_data = self.recv_all(self.socket, 8)
                if not frame_count_data:
                    raise ConnectionError("Failed to receive frame count")
                    
                frame_count = struct.unpack('!Q', frame_count_data)[0]
                if frame_count == 0:
                    # No frames available
                    return False
                
                # Receive all frames
                for _ in range(frame_count):
                    self._receive_buffer_frame()
                    
                return True
                    
        except Exception as e:
            self._handle_connection_error(f"Error requesting frame range: {str(e)}")
            return False
            
    def request_latest_n_frames(self, n):
        """
        Request the latest N frames from a buffer server.
        
        Args:
            n: Number of frames to request
            
        Returns:
            Boolean indicating success
        """
        if not self.connected or self.server_type != self.SERVER_TYPE_BUFFER:
            return False
            
        # Ensure n is at least 1
        n = max(1, n) 
            
        try:
            with self.socket_lock:
                # Send command 3 (Get latest N frames)
                self.socket.sendall(struct.pack('!I', 3))
                
                # Send count
                self.socket.sendall(struct.pack('!q', n))
                
                # Receive frame count
                frame_count_data = self.recv_all(self.socket, 8)
                if not frame_count_data:
                    raise ConnectionError("Failed to receive frame count")
                    
                frame_count = struct.unpack('!Q', frame_count_data)[0]
                if frame_count == 0:
                    # No frames available
                    return False
                
                # Receive all frames
                for _ in range(frame_count):
                    self._receive_buffer_frame()
                    
                return True
                    
        except Exception as e:
            self._handle_connection_error(f"Error requesting latest N frames: {str(e)}")
            return False
            
    def request_buffer_info(self):
        """
        Request buffer information from a buffer server.
        
        Returns:
            Boolean indicating success
        """
        if not self.connected or self.server_type != self.SERVER_TYPE_BUFFER:
            return False
            
        try:
            with self.socket_lock:
                # Send command 4 (Get buffer info)
                self.socket.sendall(struct.pack('!I', 4))
                
                # Receive info data
                info_data = self.recv_all(self.socket, 24)  # 3 int64 values
                if not info_data or len(info_data) < 24:
                    raise ConnectionError("Failed to receive buffer info")
                    
                min_frame, max_frame, buffer_size = struct.unpack('!qqq', info_data)
                
                # -1 indicates invalid/empty buffer
                if min_frame != -1 and max_frame != -1:
                    self.server_buffer_size = buffer_size
                    
                    # Call callback if set
                    if self.on_buffer_info_callback:
                        self.on_buffer_info_callback(min_frame, max_frame, buffer_size)
                        
                    return True
                    
        except Exception as e:
            self._handle_connection_error(f"Error requesting buffer info: {str(e)}")
            return False
            
    def _receive_buffer_frame(self):
        """
        Receive a single frame from a buffer server.
        This must be called with socket_lock already acquired.
        
        Returns:
            Boolean indicating success
        """
        # Receive header with payload length
        header = self.recv_all(self.socket, 8)
        if not header:
            raise ConnectionError("Failed to receive frame header")
            
        total_payload_len = struct.unpack('!Q', header)[0]
        
        # Make sure payload is at least big enough for count and time
        if total_payload_len < self.count_time_size:
            raise ValueError(f"Invalid payload size: {total_payload_len}")
            
        # Receive the entire payload
        payload = self.recv_all(self.socket, total_payload_len)
        if not payload:
            raise ConnectionError("Failed to receive frame payload")
            
        # Process the frame
        self._process_frame_data(payload)
        return True
        
    # --- Callback registration methods ---
    
    def register_frame_callback(self, callback):
        """Register callback for new frames"""
        self.on_frame_callback = callback
        
    def register_buffer_info_callback(self, callback):
        """Register callback for buffer info updates"""
        self.on_buffer_info_callback = callback
        
    def register_error_callback(self, callback):
        """Register callback for connection errors"""
        self.on_connection_error_callback = callback
        
    def close(self):
        """Close the stream and release resources"""
        self.connected = False
        if self.socket:
            try:
                self.socket.close()
            except:
                pass
        self.socket = None