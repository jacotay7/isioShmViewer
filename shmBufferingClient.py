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
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, pyqtSlot, QMetaObject, Q_ARG
import pyqtgraph as pg
from shm_stream import SHMStream  # Import the new SHMStream class

class SHMBufferingClient(QWidget):
    """
    Client widget that connects to a buffering server and can request
    contiguous blocks of data rather than just the latest frame.
    Uses SHMStream to handle network communication and data buffering.
    """
    # Signals for thread-safe UI updates
    frame_received = pyqtSignal(np.ndarray, int, float)
    buffer_info_received = pyqtSignal(int, int, int)
    connection_error = pyqtSignal(str)
    update_plot_title = pyqtSignal(str)  # Signal for updating the plot title
    update_controls_enabled = pyqtSignal(bool)  # Signal for enabling/disabling controls
    update_buffer_text = pyqtSignal(str)  # Signal for updating the buffer text
    
    def __init__(self, parent=None, shm_name=None, server_ip=None, custom_shape=None, port=5124):
        super(SHMBufferingClient, self).__init__(parent)
        self.shm_name = shm_name if shm_name else "default_shm"
        self.server_ip = server_ip if server_ip else "127.0.0.1"
        self.custom_shape = custom_shape
        self.port = port
        
        # Local UI state
        self.current_frame_index = -1
        self.frames_buffer = []  # Local UI buffer to store received frames
        
        # Setup UI
        self.init_ui()
        
        # Connect signals
        self.frame_received.connect(self.on_frame_received)
        self.buffer_info_received.connect(self.on_buffer_info_received)
        self.connection_error.connect(self.on_connection_error)
        self.update_plot_title.connect(self.plot_item.setTitle)
        self.update_controls_enabled.connect(self.set_controls_enabled)
        self.update_buffer_text.connect(self.buffer_text.setText)
        
        # Create the SHMStream
        self.stream = SHMStream(
            shm_name=self.shm_name,
            server_ip=self.server_ip,
            port=self.port,
            server_type=SHMStream.SERVER_TYPE_BUFFER,
            custom_shape=self.custom_shape,
            buffer_size=1000  # Larger buffer size for buffering client
        )
        
        # Register callbacks
        self.stream.register_frame_callback(self.on_stream_frame)
        self.stream.register_buffer_info_callback(self.on_stream_buffer_info)
        self.stream.register_error_callback(self.on_stream_error)
        
        # Connect to the server
        self.stream.connect()
        
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
        
        # Update controls state if we have buffer info
        if hasattr(self, 'stream') and self.stream.buffer_max_frame > 0:
            self.start_frame_spin.setMaximum(self.stream.buffer_max_frame)
            self.end_frame_spin.setMaximum(self.stream.buffer_max_frame)
            self.start_frame_spin.setValue(self.stream.buffer_min_frame)
            self.end_frame_spin.setValue(self.stream.buffer_max_frame)
        
    # Callbacks from SHMStream
    def on_stream_frame(self, frame, count, timestamp):
        """Handle frame received from SHMStream"""
        self.frame_received.emit(frame, count, timestamp)
    
    def on_stream_buffer_info(self, min_frame, max_frame, buffer_size):
        """Handle buffer info from SHMStream"""
        self.buffer_info_received.emit(min_frame, max_frame, buffer_size)
    
    def on_stream_error(self, error_message):
        """Handle error from SHMStream"""
        self.connection_error.emit(error_message)
    
    # UI Signal handlers
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
        self.update_buffer_text.emit(f"Buffer: {min_frame}-{max_frame} ({buffer_size} frames)")
        
        # Update UI
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
        self.update_plot_title.emit(f"SHM: {self.shm_name} ({self.server_ip}:{self.port}) - Error: {error}")
        self.update_controls_enabled.emit(False)
    
    # User interaction handlers
    def on_fetch_clicked(self):
        """Handle fetch button click"""
        mode = self.mode_combo.currentData()
        
        # Clear buffer
        self.frames_buffer = []
        self.current_frame_index = -1
        
        if mode == 1:  # Latest Frame
            if self.stream.request_latest_frame():
                frame, count, timestamp = self.stream.get_frame()
                if frame is not None:
                    self.frames_buffer.append((frame, count, timestamp))
                    self.current_frame_index = 0
                    self.display_frame(frame, count, timestamp)
                    self.frame_slider.setMaximum(0)
                    self.frame_slider.setValue(0)
        
        elif mode == 2:  # Frame Range
            start_frame = self.start_frame_spin.value()
            end_frame = self.end_frame_spin.value()
            max_frames = self.max_frames_spin.value()
            
            if self.stream.request_frame_range(start_frame, end_frame, max_frames):
                # Get contiguous buffer from stream
                frames_array, counts_array, times_array = self.stream.get_contiguous_buffer(
                    start_frame, end_frame, max_frames
                )
                
                if frames_array is not None:
                    # Convert to list of tuples for UI
                    for i in range(len(counts_array)):
                        if not np.isnan(frames_array[i]).all():  # Skip frames that are all NaN
                            self.frames_buffer.append((frames_array[i], counts_array[i], times_array[i]))
                    
                    if self.frames_buffer:
                        self.current_frame_index = 0
                        self.display_current_frame()
                        self.frame_slider.setMaximum(len(self.frames_buffer) - 1)
                        self.frame_slider.setValue(0)
        
        elif mode == 3:  # Latest N Frames
            n = self.max_frames_spin.value()
            
            if self.stream.request_latest_n_frames(n):
                # Get latest frames from stream buffer
                max_count = self.stream.buffer_max_frame
                min_count = max(self.stream.buffer_min_frame, max_count - n + 1)
                
                frames_array, counts_array, times_array = self.stream.get_contiguous_buffer(
                    min_count, max_count
                )
                
                if frames_array is not None:
                    # Convert to list of tuples for UI
                    for i in range(len(counts_array)):
                        if not np.isnan(frames_array[i]).all():  # Skip frames that are all NaN
                            self.frames_buffer.append((frames_array[i], counts_array[i], times_array[i]))
                    
                    if self.frames_buffer:
                        self.current_frame_index = len(self.frames_buffer) - 1
                        self.display_current_frame()
                        self.frame_slider.setMaximum(len(self.frames_buffer) - 1)
                        self.frame_slider.setValue(self.current_frame_index)
    
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
        
    def __del__(self):
        """Clean up resources when the widget is destroyed"""
        if hasattr(self, 'stream'):
            self.stream.close()


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