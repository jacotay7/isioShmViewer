import socket
import threading
import time
import numpy as np
import struct
import ImageStreamIOWrap as ISIO
import argparse

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
        shape = shm_obj.md.size  # e.g., [1024, 1024]
        total = np.prod(shape)
        return total * np.dtype(np.float32).itemsize

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

def handle_client(conn, addr):
    """
    Handle client connection:
    1. Receive requested shared memory name
    2. Verify shared memory exists
    3. Stream frames with bandwidth management
    """
    try:
        # Negotiate shared memory name
        shm_name = conn.recv(256).decode().strip('\x00')
        shm_obj = ImageStreamIO.open_shm(shm_name)
        
        if not shm_obj:
            conn.send(struct.pack('!Q', 0))  # Send invalid size
            return

        frame_size = ImageStreamIO.get_data_size(shm_obj)
        conn.send(struct.pack('!Q', frame_size))  # Send valid frame size
        
        # Send the image shape
        shape = shm_obj.md.size  # Assuming this returns a tuple like (1024, 1024)
        # Collapse any axes which are 1
        shape = tuple(dim for dim in shape if dim > 1)
        # If there's only one axis, add a dummy axis of size 1
        if len(shape) == 1:
            shape = (shape[0], 1)
        conn.send(struct.pack('!II', *shape))  # Send the shape as two 4-byte integers
        
        bw_manager = BandwidthManager(10 * 1024 * 1024)  # 10 MB/s default cap

        while True:
            start_time = time.time()
            
            # Get frame data
            frame = ImageStreamIO.read_shm(shm_obj)
            serialized = frame.tobytes()
            
            # Send frame with header
            header = struct.pack('!Q', len(serialized))
            conn.sendall(header + serialized)
            
            # Bandwidth management
            sleep_time = bw_manager.update(len(serialized))
            if sleep_time > 0:
                time.sleep(sleep_time)
            
            # Reset bandwidth counter every second
            if time.time() - bw_manager.start_time > 1:
                bw_manager.reset()

    except Exception as e:
        print(f"Client {addr} disconnected: {str(e)}")
    finally:
        conn.close()

def start_server(host='127.0.0.1', port=5123):
    """Start streaming server with bandwidth management"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind((host, port))
        s.listen()
        print(f"Server listening on {host}:{port}")

        while True:
            conn, addr = s.accept()
            print(f"New connection from {addr}")
            client_thread = threading.Thread(
                target=handle_client, 
                args=(conn, addr)
            )
            client_thread.start()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='SHM Viewer Server')
    parser.add_argument('--ip', type=str, default='127.0.0.1',
                        help='IP address to bind the server to (default: 127.0.0.1)')
    parser.add_argument('--port', type=int, default=5123,
                        help='Port to bind the server to (default: 5123)')
    
    args = parser.parse_args()
    start_server(host=args.ip, port=args.port)