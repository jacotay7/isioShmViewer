import numpy as np
import logging
import ImageStreamIOWrap as ISIO

# Setup logger
logger = logging.getLogger(__name__)

class ImageStreamIO:
    """
    Wrapper class for ImageStreamIO operations.
    Provides methods to interact with shared memory objects.
    """
    
    @staticmethod
    def open_shm(shm_name):
        """
        Open an existing shared memory image stream.
        Returns the image object if successful, or None if the open fails.
        
        Args:
            shm_name (str): Name of the shared memory object
            
        Returns:
            ISIO.Image or None: The image object if successful, None otherwise
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
        
        Args:
            shm_obj (ISIO.Image): The shared memory image object
            
        Returns:
            tuple: (numpy.ndarray, int, datetime) - The frame data, counter value, and timestamp
        """
        data = shm_obj.copy()
        md = shm_obj.md
        cnt = md.cnt0
        t = md.lastaccesstime
        # Ensure it's a numpy array
        if not isinstance(data, np.ndarray):
            data = np.array(data)
        # Squeeze out any dimensions that are of size 1
        return np.squeeze(data), cnt, t

    @staticmethod
    def get_data_size(shm_obj):
        """
        Calculate the size in bytes of the shared memory image.
        Assumes the image data is of type FLOAT (np.float32).
        
        Args:
            shm_obj (ISIO.Image): The shared memory image object
            
        Returns:
            int: Size in bytes
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
    
    @staticmethod
    def get_shape(shm_obj):
        """
        Get the shape of the shared memory object.
        Filters out dimensions of size 1 and ensures 2D shape.
        
        Args:
            shm_obj (ISIO.Image): The shared memory image object
            
        Returns:
            tuple: 2D shape of the data
        """
        if not hasattr(shm_obj, 'md') or not hasattr(shm_obj.md, 'size') or not shm_obj.md.size:
            logger.error("SHM object metadata or size is invalid.")
            return (1, 1)
            
        shape = shm_obj.md.size
        # Filter out dimensions of size 1
        shape = tuple(dim for dim in shape if dim > 1)
        
        # Ensure at least 2D shape
        if len(shape) == 1:
            shape = (shape[0], 1)
        elif not shape:
            shape = (1, 1)
            logger.warning(f"SHM resulted in an empty shape after filtering. Using (1, 1).")
            
        return shape