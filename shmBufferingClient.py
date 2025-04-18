#!/usr/bin/env python3
import sys
import socket
import struct
import threading
import numpy as np
import time
from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QMenu,
    QVBoxLayout,
    QHBoxLayout,
    QWidget,
    QSplitter,
    QInputDialog,
    QPushButton,
    QComboBox,
    QLabel,
    QSpinBox,
    QSlider,
    QGroupBox
)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, pyqtSlot
import pyqtgraph as pg

class SHMBufferingClient(QWidget):
    """
    Client widget that connects to a buffering server and can request
    contiguous blocks of data rather than just the latest frame.
    """
    # Signals for thread-safe UI updates
    frame_received = pyqtSignal(np.ndarray, int, float)
    buffer_info_received = pyqtSignal(int, int, int)
    connection_error = pyqtSignal(str)

    def __init__(self, parent=None, shm_name=None, server_ip=None, custom_shape=None, port=5124):
        super(SHMBufferingClient, self).__init__(parent)
        self.shm_name = shm_name if shm_name else "default_shm"
        self.server_ip = server_ip if server_ip else "127.0.0.1"
        self.custom_shape = custom_shape
        self.port = port
        self.connected = False
        self.socket = None
        self.socket_lock = threading.Lock()
        self.frames_buffer = []  # Local buffer to store received frames
        self.current_frame_index = -1

        # Buffer metadata
        self.buffer_min_frame = -1
        self.buffer_max_frame = -1
        self.buffer_size = 0

        # Thread for background buffer info updates
        self.buffer_info_thread = None
        self.buffer_info_running = False

        # Setup UI
        self.init_ui()

        # Connect signals
        self.frame_received.connect(self.on_frame_received)
        self.buffer_info_received.connect(self.on_buffer_info_received)
        self.connection_error.connect(self.on_connection_error)

        # Start connection thread
        self.connect_to_server()

    def init_ui(self):
        """Initialize the UI components"""
        # Main layout
        layout = QVBoxLayout()
        self.setLayout(layout)

        # Graph layout for image display
        self.graphics_widget = pg.GraphicsLayoutWidget()
        layout.addWidget(self.graphics_widget)

        # Create a PlotItem and add an ImageItem for efficient image rendering
        self.plot_item = self.graphics_widget.addPlot()
        self.plot_item.setTitle(f"SHM: {self.shm_name} ({self.server_ip}:{self.port}) - Connecting...")
        self.plot_item.hideAxis('left')  # Hide the y-axis
        self.plot_item.hideAxis('bottom')  # Hide the x-axis
        self.image_item = pg.ImageItem()
        self.plot_item.addItem(self.image_item)

        # Create TextItems for overlay information
        text_color = (200, 200, 200)  # Light gray color for text
        self.count_text = pg.TextItem(color=text_color, anchor=(0, 0))
        self.time_text = pg.TextItem(color=text_color, anchor=(0, 1))
        self.buffer_text = pg.TextItem(color=text_color, anchor=(0, 2))
        self.plot_item.addItem(self.count_text)
        self.plot_item.addItem(self.time_text)
        self.plot_item.addItem(self.buffer_text)

        # Position text items (relative to view coordinates)
        self.count_text.setPos(10, 10)  # Top-left corner
        self.time_text.setPos(10, 30)  # Below count text
        self.buffer_text.setPos(10, 50)  # Below time text

        # Controls group
        controls_group = QGroupBox("Data Controls")
        controls_layout = QVBoxLayout()
        controls_group.setLayout(controls_layout)

        # Mode selection
        mode_layout = QHBoxLayout()
        mode_label = QLabel("Mode:")
        self.mode_combo = QComboBox()
        self.mode_combo.addItem("Latest Frame", 1)
        self.mode_combo.addItem("Frame Range", 2)
        self.mode_combo.addItem("Latest N Frames", 3)
        self.mode_combo.currentIndexChanged.connect(self.on_mode_changed)
        mode_layout.addWidget(mode_label)
        mode_layout.addWidget(self.mode_combo)
        controls_layout.addLayout(mode_layout)

        # Frame slider for navigating buffered frames
        slider_layout = QHBoxLayout()
        slider_label = QLabel("Frame:")
        self.frame_slider = QSlider(Qt.Horizontal)
        self.frame_slider.setMinimum(0)
        self.frame_slider.setMaximum(0)  # Will be updated when we get buffer info
        self.frame_slider.valueChanged.connect(self.on_slider_changed)
        slider_layout.addWidget(slider_label)
        slider_layout.addWidget(self.frame_slider)
        controls_layout.addLayout(slider_layout)

        # Frame range controls
        range_layout = QHBoxLayout()
        start_label = QLabel("Start:")
        self.start_frame_spin = QSpinBox()
        self.start_frame_spin.setMinimum(0)
        self.start_frame_spin.setMaximum(999999)
        end_label = QLabel("End:")
        self.end_frame_spin = QSpinBox()
        self.end_frame_spin.setMinimum(0)
        self.end_frame_spin.setMaximum(999999)
        max_label = QLabel("Max:")
        self.max_frames_spin = QSpinBox()
        self.max_frames_spin.setMinimum(1)
        self.max_frames_spin.setMaximum(1000)
        self.max_frames_spin.setValue(50)  # Default value
        range_layout.addWidget(start_label)
        range_layout.addWidget(self.start_frame_spin)
        range_layout.addWidget(end_label)
        range_layout.addWidget(self.end_frame_spin)
        range_layout.addWidget(max_label)
        range_layout.addWidget(self.max_frames_spin)
        controls_layout.addLayout(range_layout)

        # Buttons
        button_layout = QHBoxLayout()
        self.fetch_button = QPushButton("Fetch Data")
        self.fetch_button.clicked.connect(self.on_fetch_clicked)
        self.prev_button = QPushButton("Previous")
        self.prev_button.clicked.connect(self.on_prev_clicked)
        self.next_button = QPushButton("Next")
        self.next_button.clicked.connect(self.on_next_clicked)
        button_layout.addWidget(self.fetch_button)
        button_layout.addWidget(self.prev_button)
        button_layout.addWidget(self.next_button)
        controls_layout.addLayout(button_layout)

        layout.addWidget(controls_group)

        # Initially disable controls until connected
        self.set_controls_enabled(False)

        # Update mode visibility
        self.on_mode_changed(0)

    def set_controls_enabled(self, enabled):
        """Enable or disable all controls"""
        self.mode_combo.setEnabled(enabled)
        self.frame_slider.setEnabled(enabled)
        self.fetch_button.setEnabled(enabled)
        self.prev_button.setEnabled(enabled and self.current_frame_index > 0)
        self.next_button.setEnabled(enabled and self.current_frame_index < len(self.frames_buffer) - 1)
        self.start_frame_spin.setEnabled(enabled)
        self.end_frame_spin.setEnabled(enabled)
        self.max_frames_spin.setEnabled(enabled)

    def on_mode_changed(self, index):
        """Handle mode combo box changes"""
        mode = self.mode_combo.currentData()
        
        # Show/hide controls based on mode
        range_controls_visible = (mode == 2)  # Frame Range mode
        self.start_frame_spin.setVisible(range_controls_visible)
        self.end_frame_spin.setVisible(range_controls_visible)
        self.max_frames_spin.setVisible(range_controls_visible)
        
        # Update controls state if we have buffer info
        if self.buffer_max_frame > 0:
            self.start_frame_spin.setMaximum(self.buffer_max_frame)
            self.end_frame_spin.setMaximum(self.buffer_max_frame)
            self.start_frame_spin.setValue(self.buffer_min_frame)
            self.end_frame_spin.setValue(self.buffer_max_frame)

    def connect_to_server(self):
        """Connect to the buffering server in a separate thread"""
        threading.Thread(target=self._connect_thread, daemon=True).start()

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
            
            # Receive frame size
            data = self.recv_all(self.socket, 8)
            if not data:
                self.connection_error.emit("Failed to receive frame size from server")
                return
                
            frame_size = struct.unpack('!Q', data)[0]
            if frame_size == 0:
                self.connection_error.emit(f"Invalid SHM: {self.shm_name}")
                return
                
            # Receive shape
            shape_data = self.recv_all(self.socket, 8)
            if not shape_data:
                self.connection_error.emit("Failed to receive shape from server")
                return
                
            self.image_shape = struct.unpack('!II', shape_data)
            
            # Reset timeout for normal operation
            self.socket.settimeout(None)
            
            # Mark as connected and start buffer info thread
            self.connected = True
            self.plot_item.setTitle(f"SHM: {self.shm_name} ({self.server_ip}:{self.port}) - Connected")
            self.set_controls_enabled(True)
            
            # Start buffer info thread
            self.buffer_info_running = True
            self.buffer_info_thread = threading.Thread(target=self._buffer_info_thread, daemon=True)
            self.buffer_info_thread.start()
            
            # Get latest frame to start
            self.request_latest_frame()
            
        except socket.timeout:
            self.connection_error.emit(f"Connection timed out: {self.server_ip}:{self.port}")
        except ConnectionRefusedError:
            self.connection_error.emit(f"Connection refused: {self.server_ip}:{self.port}")
        except Exception as e:
            self.connection_error.emit(f"Connection error: {str(e)}")

    def _buffer_info_thread(self):
        """Thread for periodically updating buffer information"""
        while self.buffer_info_running and self.connected:
            try:
                self.request_buffer_info()
                time.sleep(1)  # Update every second
            except Exception as e:
                self.connection_error.emit(f"Buffer info error: {str(e)}")
                break

    def recv_all(self, sock, n):
        """Helper function to receive exactly n bytes from a socket."""
        data = b''
        while len(data) < n:
            packet = sock.recv(n - len(data))
            if not packet:
                return None
            data += packet
        return data

    def request_latest_frame(self):
        """Request the latest frame from the server"""
        if not self.connected:
            return
            
        try:
            with self.socket_lock:
                # Send command 1 (Get latest frame)
                self.socket.sendall(struct.pack('!I', 1))
                
                # Receive frame count
                frame_count_data = self.recv_all(self.socket, 8)
                if not frame_count_data:
                    raise Exception("Failed to receive frame count")
                    
                frame_count = struct.unpack('!Q', frame_count_data)[0]
                if frame_count == 0:
                    # No frames available
                    return
                
                # We know we're getting exactly one frame
                self.receive_frame()
                    
        except Exception as e:
            self.connection_error.emit(f"Error requesting latest frame: {str(e)}")
            self.connected = False

    def request_frame_range(self, start_frame, end_frame, max_frames):
        """Request a range of frames from the server"""
        if not self.connected:
            return
            
        try:
            with self.socket_lock:
                # Send command 2 (Get frame range)
                self.socket.sendall(struct.pack('!I', 2))
                
                # Send range parameters
                self.socket.sendall(struct.pack('!qqq', start_frame, end_frame, max_frames))
                
                # Receive frame count
                frame_count_data = self.recv_all(self.socket, 8)
                if not frame_count_data:
                    raise Exception("Failed to receive frame count")
                    
                frame_count = struct.unpack('!Q', frame_count_data)[0]
                if frame_count == 0:
                    # No frames available
                    return
                
                # Clear local buffer
                self.frames_buffer = []
                self.current_frame_index = -1
                
                # Receive all frames
                for _ in range(frame_count):
                    self.receive_frame()
                    
                # Update controls
                if self.frames_buffer:
                    self.current_frame_index = 0
                    self.display_current_frame()
                    self.frame_slider.setMaximum(len(self.frames_buffer) - 1)
                    self.frame_slider.setValue(0)
                    
        except Exception as e:
            self.connection_error.emit(f"Error requesting frame range: {str(e)}")
            self.connected = False

    def request_latest_n_frames(self, n):
        """Request the latest N frames from the server"""
        if not self.connected:
            return
            
        try:
            with self.socket_lock:
                # Send command 3 (Get latest N frames)
                self.socket.sendall(struct.pack('!I', 3))
                
                # Send count
                self.socket.sendall(struct.pack('!q', n))
                
                # Receive frame count
                frame_count_data = self.recv_all(self.socket, 8)
                if not frame_count_data:
                    raise Exception("Failed to receive frame count")
                    
                frame_count = struct.unpack('!Q', frame_count_data)[0]
                if frame_count == 0:
                    # No frames available
                    return
                
                # Clear local buffer
                self.frames_buffer = []
                self.current_frame_index = -1
                
                # Receive all frames
                for _ in range(frame_count):
                    self.receive_frame()
                    
                # Update controls to show the latest (last) frame
                if self.frames_buffer:
                    self.current_frame_index = len(self.frames_buffer) - 1
                    self.display_current_frame()
                    self.frame_slider.setMaximum(len(self.frames_buffer) - 1)
                    self.frame_slider.setValue(self.current_frame_index)
                    
        except Exception as e:
            self.connection_error.emit(f"Error requesting latest N frames: {str(e)}")
            self.connected = False

    def request_buffer_info(self):
        """Request buffer information from the server"""
        if not self.connected:
            return
            
        try:
            with self.socket_lock:
                # Send command 4 (Get buffer info)
                self.socket.sendall(struct.pack('!I', 4))
                
                # Receive info data
                info_data = self.recv_all(self.socket, 24)  # 3 int64 values
                if not info_data or len(info_data) < 24:
                    raise Exception("Failed to receive buffer info")
                    
                min_frame, max_frame, buffer_size = struct.unpack('!qqq', info_data)
                
                # -1 indicates invalid/empty buffer
                if min_frame != -1 and max_frame != -1:
                    self.buffer_info_received.emit(min_frame, max_frame, buffer_size)
                    
        except Exception as e:
            self.connection_error.emit(f"Error requesting buffer info: {str(e)}")
            self.connected = False

    def receive_frame(self):
        """
        Receive a single frame from the server and add it to the buffer
        This must be called with socket_lock already acquired
        """
        # Receive header with payload length
        header = self.recv_all(self.socket, 8)
        if not header:
            raise Exception("Failed to receive frame header")
            
        total_payload_len = struct.unpack('!Q', header)[0]
        
        # Define struct formats for count (int64) and time (float64)
        count_time_format = '!qd'
        count_time_size = struct.calcsize(count_time_format)
        
        # Make sure payload is at least big enough for count and time
        if total_payload_len < count_time_size:
            raise Exception(f"Invalid payload size: {total_payload_len}")
            
        # Receive the entire payload
        payload = self.recv_all(self.socket, total_payload_len)
        if not payload:
            raise Exception("Failed to receive frame payload")
            
        # Extract count and time from the beginning
        frame_count, frame_time = struct.unpack(count_time_format, payload[:count_time_size])
        
        # The rest is the frame data
        frame_bytes = payload[count_time_size:]
        
        # Convert bytes to numpy array and reshape it
        try:
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
            
            # Add to buffer and emit signal
            self.frames_buffer.append((frame, frame_count, frame_time))
            self.frame_received.emit(frame, frame_count, frame_time)
            
        except Exception as e:
            raise Exception(f"Error processing frame data: {str(e)}")

    def on_frame_received(self, frame, frame_count, frame_time):
        """Handle received frame in UI thread"""
        # If we only have one frame, display it
        if len(self.frames_buffer) == 1:
            self.current_frame_index = 0
            self.display_frame(frame, frame_count, frame_time)
            self.frame_slider.setMaximum(0)
            self.frame_slider.setValue(0)

    def on_buffer_info_received(self, min_frame, max_frame, buffer_size):
        """Handle buffer info updates in UI thread"""
        self.buffer_min_frame = min_frame
        self.buffer_max_frame = max_frame
        self.buffer_size = buffer_size
        
        # Update UI
        self.buffer_text.setText(f"Buffer: {min_frame}-{max_frame} ({buffer_size} frames)")
        
        # Update controls
        self.start_frame_spin.setMinimum(min_frame)
        self.start_frame_spin.setMaximum(max_frame)
        self.end_frame_spin.setMinimum(min_frame)
        self.end_frame_spin.setMaximum(max_frame)
        
        if self.start_frame_spin.value() < min_frame:
            self.start_frame_spin.setValue(min_frame)
        if self.end_frame_spin.value() > max_frame:
            self.end_frame_spin.setValue(max_frame)

    def on_connection_error(self, error):
        """Handle connection errors in UI thread"""
        self.plot_item.setTitle(f"SHM: {self.shm_name} ({self.server_ip}:{self.port}) - Error: {error}")
        self.connected = False
        self.set_controls_enabled(False)
        self.buffer_info_running = False

    def on_fetch_clicked(self):
        """Handle fetch button click"""
        mode = self.mode_combo.currentData()
        
        # Clear buffer and request new data based on mode
        self.frames_buffer = []
        self.current_frame_index = -1
        
        if mode == 1:  # Latest Frame
            self.request_latest_frame()
        elif mode == 2:  # Frame Range
            start_frame = self.start_frame_spin.value()
            end_frame = self.end_frame_spin.value()
            max_frames = self.max_frames_spin.value()
            self.request_frame_range(start_frame, end_frame, max_frames)
        elif mode == 3:  # Latest N Frames
            n = self.max_frames_spin.value()
            self.request_latest_n_frames(n)

    def on_slider_changed(self, value):
        """Handle slider value change"""
        if 0 <= value < len(self.frames_buffer):
            self.current_frame_index = value
            self.display_current_frame()
            
            # Update prev/next buttons
            self.prev_button.setEnabled(self.current_frame_index > 0)
            self.next_button.setEnabled(self.current_frame_index < len(self.frames_buffer) - 1)

    def on_prev_clicked(self):
        """Display previous frame"""
        if self.current_frame_index > 0:
            self.current_frame_index -= 1
            self.display_current_frame()
            self.frame_slider.setValue(self.current_frame_index)

    def on_next_clicked(self):
        """Display next frame"""
        if self.current_frame_index < len(self.frames_buffer) - 1:
            self.current_frame_index += 1
            self.display_current_frame()
            self.frame_slider.setValue(self.current_frame_index)

    def display_current_frame(self):
        """Display the current frame from the buffer"""
        if 0 <= self.current_frame_index < len(self.frames_buffer):
            frame, count, timestamp = self.frames_buffer[self.current_frame_index]
            self.display_frame(frame, count, timestamp)
            
            # Update navigation controls
            self.prev_button.setEnabled(self.current_frame_index > 0)
            self.next_button.setEnabled(self.current_frame_index < len(self.frames_buffer) - 1)

    def display_frame(self, frame, count, timestamp):
        """Display a frame on the image item"""
        self.image_item.setImage(frame, autoLevels=True)
        self.count_text.setText(f"Frame: {count} ({self.current_frame_index + 1}/{len(self.frames_buffer)})")
        self.time_text.setText(f"Time: {timestamp:.3f}")

class BufferingMainWindow(QMainWindow):
    """
    The main window holds a single SHMBufferingClient widget.
    """
    def __init__(self):
        super(BufferingMainWindow, self).__init__()
        self.setWindowTitle("SHM Buffering Viewer")
        
        # Ask for server IP and port at startup
        server_input, ok = QInputDialog.getText(self, "Server IP", "Enter server IP address (IP:PORT):", text="127.0.0.1:5124")
        if not ok or not server_input:
            server_input = "127.0.0.1:5124"
            
        # Parse server_input to get IP and port
        server_ip = "127.0.0.1"
        port = 5124
        if ":" in server_input:
            server_ip, port_str = server_input.rsplit(":", 1)
            try:
                port = int(port_str)
            except ValueError:
                print(f"Invalid port: {port_str}. Using default port 5124.")
                port = 5124
        else:
            server_ip = server_input
        
        # Ask for SHM name
        shm_name, ok = QInputDialog.getText(self, "SHM Name", "Enter SHM name:")
        if not ok or not shm_name:
            shm_name = "default_shm"
        
        # Ask for custom shape
        use_custom_shape, ok = QInputDialog.getText(
            self, 
            "Custom Shape", 
            "Enter custom shape (e.g., '512,512') or leave empty for default:",
            text=""
        )
        
        custom_shape = None
        if ok and use_custom_shape:
            try:
                # Parse the shape string into a tuple of integers
                shape_values = [int(x.strip()) for x in use_custom_shape.split(',')]
                if len(shape_values) >= 2:
                    custom_shape = tuple(shape_values)
                    print(f"Using custom shape: {custom_shape}")
            except ValueError:
                print("Invalid shape format. Using default shape.")
        
        # Create client widget
        self.client_widget = SHMBufferingClient(
            self,
            shm_name=shm_name,
            server_ip=server_ip,
            custom_shape=custom_shape,
            port=port
        )
        
        self.setCentralWidget(self.client_widget)
        self.resize(800, 600)


def main():
    app = QApplication(sys.argv)
    window = BufferingMainWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()