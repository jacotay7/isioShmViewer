#!/usr/bin/env python
import time
import numpy as np
from isio_wrapper import ImageStreamIO
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
        img = ImageStreamIO.create_shm(shm_name, shape, np.float32)
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
        ("rtc_dtt_controller_commands_clipped", (2, 1))
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
