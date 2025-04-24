import struct
import numpy as np
import logging
from datetime import datetime


# Setup logger
logger = logging.getLogger(__name__)

def read_shm_capture_file(filename):
    """
    Read frames from a recorded SHM capture file.
    
    Args:
        filename (str): Path to the capture file.
        
    Returns:
        dict: A dictionary containing metadata and frames.
            metadata: Dictionary with file header information.
            frames: List of dictionaries, each with 'frame', 'frame_number', and 'timestamp'.
            
    Usage:
        ```python
        # Read a capture file
        data = read_shm_capture_file("path/to/capture_file.bin")
        
        # Access metadata
        print(f"SHM name: {data['metadata']['shm_name']}")
        print(f"Frame shape: ({data['metadata']['height']}, {data['metadata']['width']})")
        print(f"Recorded at: {data['metadata']['target_fps']} FPS")
        
        # Access frames
        for i, frame_data in enumerate(data['frames']):
            frame = frame_data['frame']  # NumPy array
            frame_number = frame_data['frame_number']
            timestamp = frame_data['timestamp']
            
            # Process frame (e.g., display, analyze, etc.)
            print(f"Frame {i}: number={frame_number}, timestamp={timestamp}, shape={frame.shape}")
            
            # Example: Display frame (requires matplotlib)
            # import matplotlib.pyplot as plt
            # plt.figure()
            # plt.imshow(frame)
            # plt.title(f"Frame {frame_number}")
            # plt.colorbar()
            # plt.show()
        ```
    """
    # Map dtype codes back to numpy dtypes
    dtype_code_to_dtype = {
        1: np.dtype('float32'),
        2: np.dtype('float64'),
        3: np.dtype('uint8'),
        4: np.dtype('uint16'),
        5: np.dtype('int16'),
        6: np.dtype('int32'),
        7: np.dtype('uint32'),
        0: np.dtype('float32'),  # Default
    }
    
    try:
        with open(filename, 'rb') as f:
            # Read magic number
            magic = f.read(7)
            if magic != b'SHMCAPT':
                raise ValueError(f"Invalid magic number in file {filename}")
            
            # Read header
            version = struct.unpack('B', f.read(1))[0]
            shm_name_len = struct.unpack('B', f.read(1))[0]
            shm_name = f.read(shm_name_len).decode('utf-8')
            timestamp = struct.unpack('!q', f.read(8))[0]
            target_fps = struct.unpack('!f', f.read(4))[0]
            height, width = struct.unpack('!II', f.read(8))
            dtype_code, dtype_size = struct.unpack('BB', f.read(2))
            
            # Get numpy dtype
            dtype = dtype_code_to_dtype.get(dtype_code)
            if not dtype:
                logger.warning(f"Unknown dtype code {dtype_code}, defaulting to float32")
                dtype = np.dtype('float32')
            
            # Create metadata dictionary
            metadata = {
                'version': version,
                'shm_name': shm_name,
                'timestamp': timestamp,
                'timestamp_str': datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S'),
                'target_fps': target_fps,
                'height': height,
                'width': width,
                'dtype': dtype,
            }
            
            # Calculate frame data size
            frame_data_size = height * width * dtype.itemsize
            
            # Read frames
            frames = []
            while True:
                # Try to read frame metadata
                frame_metadata = f.read(16)  # 8 bytes for frame number, 8 bytes for timestamp
                if len(frame_metadata) < 16:
                    # End of file
                    break
                    
                frame_number, timestamp = struct.unpack('!qd', frame_metadata)
                
                # Read frame data
                frame_bytes = f.read(frame_data_size)
                if len(frame_bytes) < frame_data_size:
                    logger.warning(f"Incomplete frame data for frame {frame_number}")
                    break
                    
                # Convert to numpy array
                frame = np.frombuffer(frame_bytes, dtype=dtype).reshape(height, width)
                
                # Add to frames list
                frames.append({
                    'frame': frame,
                    'frame_number': frame_number,
                    'timestamp': timestamp,
                })
                
            return {
                'metadata': metadata,
                'frames': frames,
            }
            
    except Exception as e:
        logger.error(f"Error reading SHM capture file: {str(e)}")
        raise

# %% Read the capture file
data = read_shm_capture_file('rtc_dtt_controller_commands_20250424_000559.bin')

# Get metadata
metadata = data['metadata']
print(f"SHM Name: {metadata['shm_name']}")
print(f"Frame Shape: {metadata['height']} x {metadata['width']}")
print(f"Recorded at: {metadata['target_fps']} FPS")
print(f"Capture time: {metadata['timestamp_str']}")

# Process frames
frames = data['frames']
print(f"Total frames: {len(frames)}")

# %%Access a specific frame
frame_data = frames[0]  # First frame
frame = frame_data['frame']  # NumPy array
frame_number = frame_data['frame_number']
timestamp = frame_data['timestamp']

# Example: Display frame with matplotlib
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 8))
plt.imshow(frame)
plt.colorbar()
plt.title(f"Frame {frame_number} (timestamp: {timestamp:.6f})")
plt.show()

#%%