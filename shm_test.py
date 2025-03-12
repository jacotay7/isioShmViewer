#!/usr/bin/env python
import time
import numpy as np
import ImageStreamIOWrap as ISIO

def create_image(name, shape, dtype):
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

def randomize_img(img):
    test_frame = np.array(img)
    # Create a random frame
    for i in range(test_frame.shape[0]):
        for j in range(test_frame.shape[1]):
            test_frame[i, j] = np.float32(np.random.rand())
    img.write(test_frame)
    return

import threading
import signal
import sys

def stream_random_data(shm_name, shape):
    img = create_image(shm_name, shape, np.float32)
    print(f"Streaming random data to shared memory '{shm_name}' with shape {shape}...")
    while True:
        try:
            randomize_img(img)
        except Exception as e:
            print(f"Error writing to shared memory '{shm_name}':", e)
            break
        time.sleep(0.1)

def signal_handler(sig, frame):
    print("Exiting...")
    sys.exit(0)

def main():
    signal.signal(signal.SIGINT, signal_handler)

    # Define shared memory names and sizes
    shms = [
        ("test_shm_128x128", (128, 128)),
        ("test_shm_256x256", (256, 256)),
        ("test_shm_512x512", (512, 512))
    ]

    # Create and start threads for each shared memory
    threads = []
    for shm_name, shape in shms:
        thread = threading.Thread(target=stream_random_data, args=(shm_name, shape))
        thread.start()
        threads.append(thread)

    # Wait for all threads to complete
    for thread in threads:
        thread.join()

if __name__ == "__main__":
    main()
