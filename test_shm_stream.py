#!/usr/bin/env python3
"""
Test script for the SHMStream class.
Measures the time it takes to fill a buffer with 100 frames and reports on missing frames.
"""

#%%
import sys
import time
import numpy as np
import argparse
from shm_stream import SHMStream
from fakeShmStream import FakeShmStreamer

    
#%%

shm_name = "rtc_xinetics_dm_controller_commands_clipped"
buffer_size = 1000
server_ip = '127.0.0.1'
port = 5124
custom_shape = None
verbose = False

#%%
# Create SHM stream
print(f"Connecting to {server_ip}:{port}, SHM: {shm_name}")
stream = SHMStream(
    shm_name=shm_name,
    server_ip=server_ip,
    port=port,
    server_type=SHMStream.SERVER_TYPE_BUFFER,
    custom_shape=custom_shape,
    buffer_size=buffer_size
)
# Connect to the server
stream.connect()
    
# Make sure we're connected before starting the timer
time.sleep(1.0)
if not stream.connected:
    print(f"Failed to connect to server: {stream.connection_error}")

#%%
# Get the contiguous buffer with missing frames as NaNs
frames_array, counts_array, times_array = stream.get_contiguous_buffer()

# %%
frame_total = np.max(counts_array) - np.min(counts_array) + 1
print(len(counts_array), "frames received")
print(frame_total-len(counts_array), "frames missing")


# %%
