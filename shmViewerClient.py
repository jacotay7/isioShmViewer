import sys
import socket
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

class SHMPlotWidget(QWidget):
    """
    A pyqtgraph-based widget that displays a frame streamed from the server.
    It connects to the server, sends the SHM name, and continuously receives
    frames (expected to be 1024x1024 float32 arrays).
    """
    def __init__(self, parent=None, shm_name=None, server_ip=None):
        super(SHMPlotWidget, self).__init__(parent)
        self.shm_name = shm_name if shm_name else "default_shm"
        self.server_ip = server_ip if server_ip else "127.0.0.1"
        self.latest_frame = None

        # Set up layout and pyqtgraph graphics widget
        self.layout = QVBoxLayout()
        self.setLayout(self.layout)
        self.graphics_widget = pg.GraphicsLayoutWidget()
        self.layout.addWidget(self.graphics_widget)

        # Create a PlotItem and add an ImageItem for efficient image rendering
        self.plot_item = self.graphics_widget.addPlot()
        self.plot_item.setTitle(f"SHM: {self.shm_name} ({self.server_ip})")
        self.plot_item.hideAxis('left')  # Hide the y-axis
        self.plot_item.hideAxis('bottom')  # Hide the x-axis
        self.image_item = pg.ImageItem()
        self.plot_item.addItem(self.image_item)

        # Start the background thread to receive data from the server
        self.socket_thread = threading.Thread(target=self.receive_data, daemon=True)
        self.socket_thread.start()

        # Use a QTimer to update the image periodically
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_plot)
        self.timer.start(50)  # update every 50 ms (adjust as needed)

    def recv_all(self, sock, n):
        """Helper function to receive exactly n bytes from a socket."""
        data = b''
        while len(data) < n:
            packet = sock.recv(n - len(data))
            if not packet:
                return None
            data += packet
        return data

    def receive_data(self):
        """
        Connect to the server and continuously receive frames.
        Protocol:
          1. Send the SHM name padded to 256 bytes.
          2. Receive an 8-byte frame size (0 indicates failure).
          3. Receive the image shape as a tuple.
          4. In a loop, receive an 8-byte header with the length of the upcoming frame,
             then receive the frame bytes.
        """
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.connect((self.server_ip, 5123))
                # Send the SHM name padded to 256 bytes
                name_bytes = self.shm_name.encode()
                name_padded = name_bytes.ljust(256, b'\x00')
                s.sendall(name_padded)
                
                # Receive initial frame size (8 bytes)
                data = self.recv_all(s, 8)
                if data is None:
                    print("Failed to receive frame size from server.")
                    return
                expected_frame_size = struct.unpack('!Q', data)[0]
                if expected_frame_size == 0:
                    print("Invalid SHM name. No shared memory found on server.")
                    return
                
                # Receive the image shape (e.g., 2 integers for 2D shape)
                shape_data = self.recv_all(s, 8)  # Assuming shape is sent as two 4-byte integers
                if shape_data is None:
                    print("Failed to receive image shape from server.")
                    return
                image_shape = struct.unpack('!II', shape_data)  # Adjust based on actual shape format
                
                # Now continuously receive frames
                while True:
                    # Receive header: 8 bytes indicating the length of the frame
                    header = self.recv_all(s, 8)
                    if header is None:
                        break
                    frame_length = struct.unpack('!Q', header)[0]
                    frame_bytes = self.recv_all(s, frame_length)
                    if frame_bytes is None:
                        break
                    # Convert bytes to a numpy array and reshape it using the received shape
                    frame = np.frombuffer(frame_bytes, dtype=np.float32)
                    try:
                        frame = frame.reshape(image_shape)
                    except Exception as e:
                        print("Error reshaping frame:", e)
                        continue
                    self.latest_frame = frame
        except Exception as e:
            print(f"Error in SHM socket thread connecting to {self.server_ip}:", e)

    def update_plot(self):
        """
        Update the pyqtgraph ImageItem with the latest received frame.
        """
        if self.latest_frame is not None:
            # The autoLevels=True flag adjusts the contrast automatically.
            self.image_item.setImage(self.latest_frame, autoLevels=True)

class SplitWidget(QWidget):
    """
    A widget that can display a single SHM plot or split into two new SplitWidgets.
    When a new leaf is created, the user is prompted for an SHM name unless skip_prompt is True.
    """
    def __init__(self, parent=None, shm_name=None, server_ip=None, skip_prompt=False):
        super(SplitWidget, self).__init__(parent)
        self.leaf = True   # Initially a leaf node with one plot
        self.plot_widget = None
        self.splitter = None
        self.server_ip = server_ip

        self.layout = QVBoxLayout()
        self.setLayout(self.layout)

        if not skip_prompt:
            if shm_name is None:
                self.init_leaf()
            else:
                self.plot_widget = SHMPlotWidget(self, shm_name=shm_name, server_ip=self.server_ip)
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
            server_ip, ok = QInputDialog.getText(self, "Server IP", "Enter server IP address:", text="127.0.0.1")
            if not ok or not server_ip:
                server_ip = "127.0.0.1"
            self.server_ip = server_ip

        main_window = self.window()
        if hasattr(main_window, "shm_list"):
            main_window.shm_list.append(shm_name)

        self.plot_widget = SHMPlotWidget(self, shm_name=shm_name, server_ip=self.server_ip)
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
        child1 = SplitWidget(self.splitter, server_ip=self.server_ip, skip_prompt=True)
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
        child2 = SplitWidget(self.splitter, shm_name=shm_name, server_ip=self.server_ip)

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
        
        # Ask for server IP at startup
        server_ip, ok = QInputDialog.getText(self, "Server IP", "Enter server IP address:", text="127.0.0.1")
        if not ok or not server_ip:
            server_ip = "127.0.0.1"
            
        self.split_widget = SplitWidget(self, server_ip=server_ip)
        self.setCentralWidget(self.split_widget)
        self.resize(800, 600)

def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
