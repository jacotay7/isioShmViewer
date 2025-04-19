#!/usr/bin/env python
import time
import numpy as np
import ImageStreamIOWrap as ISIO
import threading
import signal
import sys

class FakeShmStreamer:
    def __init__(self, mode="random", rate=10, data=None):
        """
        Initialize the FakeShmStreamer.

        Args:
            mode (str): The streaming mode. Options are "random", "constant", or "loop".
            data (np.ndarray, optional): A NumPy array to loop through if mode is "loop".
        """
        self.threads = []
        self.mode = mode
        self.data = data
        if self.mode == "loop" and self.data is None:
            raise ValueError("Data must be provided when mode is 'loop'.")
        self.loop_index = 0
        self.rate = rate #Hz

    def create_image(self, name, shape, dtype):
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

    def randomize_img(self, img):
        test_frame = np.array(img)
        # Create a random frame
        for i in range(test_frame.shape[0]):
            for j in range(test_frame.shape[1]):
                test_frame[i, j] = np.float32(np.random.rand())
        img.write(test_frame)
        return

    def write_constant_img(self, img):
        img.write(np.array(img))
        return

    def write_loop_img(self, img):
        """
        Write data from the user-provided NumPy array in a loop.
        """
        test_frame = self.data[self.loop_index % len(self.data)]
        img.write(test_frame)
        self.loop_index += 1
        return

    def stream_data(self, shm_name, shape, sleep=0.1):
        img = self.create_image(shm_name, shape, np.float32)
        print(f"Streaming data to shared memory '{shm_name}' with shape {shape} in mode '{self.mode}'...")
        while True:
            try:
                if self.mode == "random":
                    self.randomize_img(img)
                elif self.mode == "constant":
                    self.write_constant_img(img)
                elif self.mode == "loop":
                    self.write_loop_img(img)
            except Exception as e:
                print(f"Error writing to shared memory '{shm_name}':", e)
                break
            time.sleep(sleep)

    def start_streams(self, shms):
        for shm_name, shape in shms:
            thread = threading.Thread(target=self.stream_data, args=(shm_name, shape, 1/self.rate))
            thread.start()
            self.threads.append(thread)

    def wait_for_threads(self):
        for thread in self.threads:
            thread.join()

def signal_handler(sig, frame):
    print("Exiting...")
    sys.exit(0)

def main():
    signal.signal(signal.SIGINT, signal_handler)

    # Define shared memory names and sizes
    shms = [
        # ("rtc_ocam2k_descrambled_frame", (80, 80)),
        ("rtc_xinetics_dm_controller_commands_clipped", (349, 1)),
        # ("rtc_xinetics_residual", (354, 1)),
        # ("rtc_dtt_controller_commands_clipped", (2, 1))
    ]

    # Example usage:
    # mode = "random" for random data, "constant" for constant data, or "loop" for looping through a NumPy array
    mode = "constant"
    data = None  # Replace with a NumPy array if mode is "loop"

    streamer = FakeShmStreamer(mode=mode, data=data, rate=500)
    streamer.start_streams(shms)
    streamer.wait_for_threads()

if __name__ == "__main__":
    main()
