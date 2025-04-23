import sys
import struct
import threading
import numpy as np
from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QMenu,
    QVBoxLayout,
    QWidget,
    QSplitter,
    QInputDialog
)
from PyQt5.QtCore import Qt, QTimer
import pyqtgraph as pg
import time
from shm_stream import SHMStream  # Import the new SHMStream class

class SHMPlotWidget(QWidget):
    """
    A pyqtgraph-based widget that displays a frame streamed from the server.
    It uses SHMStream to handle the connection and protocol details.
    """
    def __init__(self, parent=None, shm_name=None, server_ip=None, custom_shape=None, port=5123):
        super(SHMPlotWidget, self).__init__(parent)
        self.shm_name = shm_name if shm_name else "default_shm"
        self.server_ip = server_ip if server_ip else "127.0.0.1"
        self.custom_shape = custom_shape
        self.port = port
        
        # FPS calculation state
        self.prev_count = -1
        self.prev_time = -1.0
        self.shm_write_fps = 0.0

        # Set up layout and pyqtgraph graphics widget
        self.layout = QVBoxLayout()
        self.setLayout(self.layout)
        self.graphics_widget = pg.GraphicsLayoutWidget()
        self.layout.addWidget(self.graphics_widget)

        # Create a PlotItem and add an ImageItem for efficient image rendering
        self.plot_item = self.graphics_widget.addPlot()
        self.plot_item.setTitle(f"SHM: {self.shm_name} ({self.server_ip}:{self.port}) - Connecting...")
        self.plot_item.hideAxis('left')  # Hide the y-axis
        self.plot_item.hideAxis('bottom')  # Hide the x-axis
        self.image_item = pg.ImageItem()
        self.plot_item.addItem(self.image_item)

        # Create TextItems for overlay
        text_color = (0, 255, 0)  # Bright green color for text
        self.count_text = pg.TextItem(color=text_color, anchor=(0, 0))
        self.time_text = pg.TextItem(color=text_color, anchor=(0, 1))
        self.fps_text = pg.TextItem(color=text_color, anchor=(0, 2))

        self.plot_item.addItem(self.count_text)
        self.plot_item.addItem(self.time_text)
        self.plot_item.addItem(self.fps_text)

        # Position text items (relative to view coordinates)
        self.count_text.setPos(0, 0)  # Top-left corner
        self.time_text.setPos(0, 1)  # Below count text
        self.fps_text.setPos(0, 2)   # Below time text

        # Create SHMStream for data handling
        self.stream = SHMStream(
            shm_name=self.shm_name,
            server_ip=self.server_ip,
            port=self.port,
            server_type=SHMStream.SERVER_TYPE_VIEWER,
            custom_shape=self.custom_shape
        )
        
        # Register callbacks for stream events
        self.stream.register_frame_callback(self.on_frame_received)
        self.stream.register_error_callback(self.on_connection_error)
        
        # Start the connection
        self.stream.connect()

        # Use a QTimer to update the image periodically
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_plot)
        self.timer.start(50)  # update every 50 ms (adjust as needed)

    def on_frame_received(self, frame, count, timestamp):
        """
        Callback for when a new frame is received from the stream.
        Updates the latest frame data.
        """
        # Calculate FPS based on SHM write times/counts
        if self.prev_time > 0 and timestamp > self.prev_time and count > self.prev_count:
            dt = timestamp - self.prev_time
            d_count = count - self.prev_count
            if dt > 0:
                self.shm_write_fps = d_count / dt
        
        # Update previous values for next calculation
        self.prev_count = count
        self.prev_time = timestamp
        
        # Note: We don't update the UI directly here since we're not in the UI thread
        # The update_plot method will handle the UI update using the latest data

    def on_connection_error(self, error_message):
        """Callback for connection errors"""
        self.plot_item.setTitle(f"SHM: {self.shm_name} ({self.server_ip}:{self.port}) - Error: {error_message}")

    def update_plot(self):
        """
        Update the pyqtgraph ImageItem with the latest received frame
        and update the title and overlay text with count, time, and FPS.
        Called by the QTimer in the UI thread.
        """
        # Get the latest frame data from the stream
        frame, count, timestamp = self.stream.get_frame()
        
        if frame is not None:
            # Update image
            self.image_item.setImage(frame, autoLevels=True)

            # Update title (optional, could remove if overlays are sufficient)
            title = f"SHM: {self.shm_name} ({self.server_ip}:{self.port})"
            self.plot_item.setTitle(title)

            # Update overlay text items
            self.count_text.setText(f"Cnt: {count}")
            self.time_text.setText(f"Time: {timestamp:.3f}")
            self.fps_text.setText(f"SHM FPS: {self.shm_write_fps:.2f}")

        else:
            # Handle case where frame is None (e.g., error or not yet received)
            self.count_text.setText("Cnt: N/A")
            self.time_text.setText("Time: N/A")
            self.fps_text.setText("SHM FPS: N/A")
            # Keep the error title if it was set
            if "Error" not in self.plot_item.titleLabel.text:
                self.plot_item.setTitle(f"SHM: {self.shm_name} ({self.server_ip}:{self.port}) - Waiting...")

    def __del__(self):
        """Clean up resources when the widget is destroyed"""
        if hasattr(self, 'stream'):
            self.stream.close()


class SplitWidget(QWidget):
    """
    A widget that can display a single SHM plot or split into two new SplitWidgets.
    When a new leaf is created, the user is prompted for an SHM name unless skip_prompt is True.
    """
    def __init__(self, parent=None, shm_name=None, server_ip=None, skip_prompt=False, custom_shape=None, port=5123):
        super(SplitWidget, self).__init__(parent)
        self.leaf = True   # Initially a leaf node with one plot
        self.plot_widget = None
        self.splitter = None
        self.server_ip = server_ip
        self.custom_shape = custom_shape
        self.port = port

        self.layout = QVBoxLayout()
        self.setLayout(self.layout)

        if not skip_prompt:
            if shm_name is None:
                self.init_leaf()
            else:
                self.plot_widget = SHMPlotWidget(self, shm_name=shm_name, server_ip=self.server_ip, custom_shape=self.custom_shape, port=self.port)
                self.layout.addWidget(self.plot_widget)
                main_window = self.window()
                if hasattr(main_window, "shm_list"):
                    main_window.shm_list.append(shm_name)
                self.setContextMenuPolicy(Qt.CustomContextMenu)
                self.customContextMenuRequested.connect(self.show_context_menu)
        # If skip_prompt is True, do nothing; expect plot_widget to be set externally.

    def init_leaf(self):
        """
        Create an SHM plot widget.
        Prompt for an SHM name and server IP, add it to the main window's list, and create the widget.
        """
        shm_name, ok = QInputDialog.getText(self, "SHM Name", "Enter SHM name:")
        if not ok or not shm_name:
            shm_name = "default_shm"
            
        if self.server_ip is None:
            server_input, ok = QInputDialog.getText(self, "Server IP", "Enter server IP address (IP:PORT):", text="127.0.0.1:5123")
            if not ok or not server_input:
                server_input = "127.0.0.1:5123"
                
            # Parse server_input to get IP and port
            if ":" in server_input:
                self.server_ip, port_str = server_input.rsplit(":", 1)
                try:
                    self.port = int(port_str)
                except ValueError:
                    print(f"Invalid port: {port_str}. Using default port 5123.")
                    self.port = 5123
            else:
                self.server_ip = server_input
                self.port = 5123
        
        # Ask if user wants to specify a custom shape
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

        main_window = self.window()
        if hasattr(main_window, "shm_list"):
            main_window.shm_list.append(shm_name)

        self.plot_widget = SHMPlotWidget(self, shm_name=shm_name, server_ip=self.server_ip, custom_shape=custom_shape, port=self.port)
        self.layout.addWidget(self.plot_widget)

        self.setContextMenuPolicy(Qt.CustomContextMenu)
        self.customContextMenuRequested.connect(self.show_context_menu)

    def show_context_menu(self, pos):
        """
        Display a context menu with options to split the widget horizontally or vertically.
        """
        if self.leaf:
            context_menu = QMenu(self)
            action_split_h = context_menu.addAction("Split Horizontally")
            action_split_v = context_menu.addAction("Split Vertically")
            action = context_menu.exec_(self.mapToGlobal(pos))
            if action == action_split_h:
                self.split(orientation=Qt.Vertical)
            elif action == action_split_v:
                self.split(orientation=Qt.Horizontal)

    def split(self, orientation=Qt.Horizontal):
        """
        Convert this leaf widget into a QSplitter that holds two new SplitWidgets.
        Only the second child will prompt for a new SHM name.
        """
        if not self.leaf:
            return

        # Create a new splitter
        self.splitter = QSplitter(orientation=orientation, parent=self)

        # Create the first child without prompting for a new name.
        child1 = SplitWidget(self.splitter, server_ip=self.server_ip, skip_prompt=True, port=self.port)
        # Transfer the existing plot widget to child1.
        child1.plot_widget = self.plot_widget
        self.plot_widget.setParent(child1)
        child1.layout.addWidget(self.plot_widget)
        child1.setContextMenuPolicy(Qt.CustomContextMenu)
        child1.customContextMenuRequested.connect(child1.show_context_menu)

        # For the second child, prompt for a new SHM name here.
        shm_name, ok = QInputDialog.getText(self, "SHM Name", "Enter SHM name:")
        if not ok or not shm_name:
            shm_name = "default_shm"
            
        # Ask if user wants to specify a custom shape
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
                
        child2 = SplitWidget(self.splitter, shm_name=shm_name, server_ip=self.server_ip, custom_shape=custom_shape, port=self.port)

        # Add both children to the splitter.
        self.splitter.addWidget(child1)
        self.splitter.addWidget(child2)

        # Clear the current layout and add the splitter.
        for i in reversed(range(self.layout.count())):
            item = self.layout.itemAt(i)
            self.layout.removeItem(item)
            if item.widget():
                item.widget().deleteLater()
        self.layout.addWidget(self.splitter)
        self.leaf = False


class MainWindow(QMainWindow):
    """
    The main window holds a single SplitWidget and a list of SHM names.
    """
    def __init__(self):
        super(MainWindow, self).__init__()
        self.setWindowTitle("Dynamic SHM Viewer")
        self.shm_list = []  # List to store SHM names from created plots
        
        # Ask for server IP and port at startup
        server_input, ok = QInputDialog.getText(self, "Server IP", "Enter server IP address (IP:PORT):", text="127.0.0.1:5123")
        if not ok or not server_input:
            server_input = "127.0.0.1:5123"
            
        # Parse server_input to get IP and port
        server_ip = "127.0.0.1"
        port = 5123
        if ":" in server_input:
            server_ip, port_str = server_input.rsplit(":", 1)
            try:
                port = int(port_str)
            except ValueError:
                print(f"Invalid port: {port_str}. Using default port 5123.")
                port = 5123
        else:
            server_ip = server_input
            
        self.split_widget = SplitWidget(self, server_ip=server_ip, port=port)
        self.setCentralWidget(self.split_widget)
        self.resize(800, 600)


def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
