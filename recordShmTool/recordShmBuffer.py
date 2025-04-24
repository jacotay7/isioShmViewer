#%%
"""
recordShm.py - Tool for recording SHM stream data to disk

This script connects to a SHM buffer server, polls the buffer at regular intervals,
and saves the collected frames to a NumPy file. It properly manages the continuous buffer
from the buffering server.
"""

import os
import sys
import time
import argparse
import numpy as np
import threading
from datetime import datetime

# Add parent directory to path to import from sibling directories
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from shm_stream import SHMStream

#%%
stream = SHMStream(
    shm_name="rtc_dtt_controller_commands_clipped",
    buffer_size=10000,
    port = 5124,
    poll_interval=0.5,
)
stream.connect()

#wait for stream.connected to be tru
while not stream.connected:
    print("Waiting for connection...")
    time.sleep(1)
print("Connected to SHM stream.")

# %%
#Monitor the length of the frame_buffer
while len(stream.frame_buffer) < stream.buffer_size:
    print(f"Buffer length: {len(stream.frame_buffer)}")
    time.sleep(1)
print("Buffer is full. Starting to record data...")
#Once it is full

# %%
