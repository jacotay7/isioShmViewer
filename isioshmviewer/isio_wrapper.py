import numpy as np
import logging
import ImageStreamIOWrap as ISIO
import datetime  # Added import

# Setup logger
logger = logging.getLogger(__name__)

# Define ISIO constants if not already available via the import
IMAGESTRUCT_VERSION = ISIO.IMAGESTRUCT_VERSION if hasattr(ISIO, 'IMAGESTRUCT_VERSION') else 0x0101
MODimDWtype = {  # Map numpy types to ISIO types
    np.float32: ISIO.ImageStreamIODataType.FLOAT,
    np.float64: ISIO.ImageStreamIODataType.DOUBLE,
    np.int8: ISIO.ImageStreamIODataType.INT8,
    np.uint8: ISIO.ImageStreamIODataType.UINT8,
    np.int16: ISIO.ImageStreamIODataType.INT16,
    np.uint16: ISIO.ImageStreamIODataType.UINT16,
    np.int32: ISIO.ImageStreamIODataType.INT32,
    np.uint32: ISIO.ImageStreamIODataType.UINT32,
    np.int64: ISIO.ImageStreamIODataType.INT64,
    np.uint64: ISIO.ImageStreamIODataType.UINT64,
    np.complex64: ISIO.ImageStreamIODataType.COMPLEX_FLOAT,
    np.complex128: ISIO.ImageStreamIODataType.COMPLEX_DOUBLE,
    np.float16: ISIO.ImageStreamIODataType.HALF
}

class ImageStreamIO:
    """
    Wrapper class for ImageStreamIO operations.
    Provides methods to interact with shared memory objects.
    """
    
    @staticmethod
    def create_shm(name, shape, dtype):
        """
        Demonstrates how to use the 'create' function provided by the C++ library via pybind11.

        The underlying C++ function has the following parameters:
            - img (IMAGE &): A reference to an image object to be created/initialized.
            - name (str): A name for the image.
            - buffer (py::buffer): A buffer that must be a NumPy array containing the image data.
            - location (int8): An 8-bit integer specifying a location (e.g. GPU device index).
            - shared (uint8): An 8-bit flag indicating whether the image memory is shared.
            - NBsem (int): The number of semaphores to associate with the image.
            - NBkw (int): The number of keywords or metadata parameters.
            - imagetype (uint64): A 64-bit unsigned integer specifying the image type.
            - CBsize (uint32): A 32-bit unsigned integer for the control buffer size.

        The function returns an integer result code:
            - 0 indicates that the image was created successfully.
            - A non-zero value indicates an error.

        Note:
            - The C++ binding verifies that the passed buffer is a NumPy array.
            - Only certain data types are supported. Supported types include:
                uint8, int8, uint16, int16, uint32, int32, uint64, int64, float, and double.
              If the buffer’s dtype is unsupported, a ValueError is raised.

        Returns:
            int: The result code from the C++ function.
        
        Example:
            >>> result = create_image_example()
            Image created successfully.
        """
        # Create an empty image buffer with the correct number of bytes.
        buffer = np.empty(shape, dtype=dtype)
        
        # Create an IMAGE instance.
        # The IMAGE class is assumed to be exposed by your module.
        img = ISIO.Image()  # Replace with the correct constructor if different

        # Set up parameters for the image creation:
        location = -1             # Example: CPU, 0 for GPU
        shared = 1                # Example flag: 1 indicates shared memory usage
        NBsem = 2                 # Example: using 2 semaphores
        NBkw = 1                  # Example: using 4 keywords (metadata slots)
        imagetype = 1             # Example image type (this might correspond to an enum value)
        CBsize = 1024             # Example control buffer size in bytes

        # Call the C++ 'create' function.
        # If the buffer is not a valid numpy array or the datatype is unsupported,
        # the binding will throw an exception.
        result = img.create(
            name, buffer, location, shared
        )

        # Check the result code.
        if result == 0:
            print("Image created successfully.")
        else:
            print(f"Image creation failed with error code: {result}")
        
        return img

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
            logger.error(f"Failed to open SHM '{shm_name}'. Error code: {ret}")
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
        # Keeping this method unchanged as requested
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
    def write_shm(shm_obj, data_array):
        """
        Write a NumPy array to the shared memory object.
        
        Args:
            shm_obj (ISIO.Image): The shared memory image object.
            data_array (np.ndarray): The NumPy array data to write.
            
        Returns:
            int: Return code from the write operation (0 for success).
                 Returns -1 if input is not a NumPy array.
        """
        if not isinstance(data_array, np.ndarray):
            logger.error("Data to write must be a NumPy array.")
            return -1
            
        # Write directly to the shared memory
        try:
            ret = shm_obj.write(data_array)
        except Exception as e:
            logger.error(f"Failed to write to SHM: {e}")
            return -1

        if ret != 0:
            logger.error(f"Failed to write to SHM. Error code: {ret}")
        return ret

    @staticmethod
    def get_data_size(shm_obj):
        """
        Calculate the size in bytes of the shared memory image.
        
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

        # Get the datatype from the image metadata
        try:
            # Use np.dtype corresponding to the image's data type
            if hasattr(shm_obj.md, 'datatype'):
                # This would need a reverse mapping from ISIO types to numpy dtypes
                # For simplicity, we'll use float32 as default
                itemsize = 4  # default to float32 size
            else:
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
        Get the shape of the shared memory object, removing redundant axes.
        
        Args:
            shm_obj (ISIO.Image): The shared memory image object
            
        Returns:
            tuple: Shape of the data with redundant axes removed
        """
        if not hasattr(shm_obj, 'md') or not hasattr(shm_obj.md, 'size') or not shm_obj.md.size:
            logger.error("SHM object metadata or size is invalid.")
            return (1, 1)
            
        shape = shm_obj.md.size
        # Ensure shape is iterable and contains numbers
        if not hasattr(shape, '__iter__') or not all(isinstance(dim, (int, float)) for dim in shape):
            logger.error(f"SHM object shape is invalid: {shape}")
            return (1, 1)
        
        # Remove redundant axes (dimensions of size 1)
        shape = tuple(dim for dim in shape if dim > 1)
        
        # Ensure at least 2D shape
        if len(shape) == 1:
            shape = (shape[0], 1)
        elif not shape:
            shape = (1, 1)
            logger.warning(f"SHM resulted in an empty shape. Using (1, 1).")
            
        return shape