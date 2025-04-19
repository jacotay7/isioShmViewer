#!/usr/bin/env python3
"""
Test script for the SHMStream class.
Connects to a buffering server, polls frames into a large client buffer,
and reports on missing frames within the collected range.
"""

#%%
import sys
import time
import numpy as np
import argparse
from shm_stream import SHMStream
# Assuming FakeShmStreamer is not needed for this test against a real server
# from fakeShmStreamer import FakeShmStreamer

    
#%%

shm_name = "rtc_xinetics_dm_controller_commands_clipped"
client_buffer_size = 10000  # Large buffer on the client side
server_ip = '127.0.0.1'
port = 5124 # Default buffering server port
custom_shape = None
verbose = False
poll_time_seconds = 10 # Time to let the client poll for frames

#%%
# Create SHM stream
print(f"Connecting to {server_ip}:{port}, SHM: {shm_name}")
print(f"Client buffer size: {client_buffer_size}")
stream = SHMStream(
    shm_name=shm_name,
    server_ip=server_ip,
    port=port,
    server_type=SHMStream.SERVER_TYPE_BUFFER,
    custom_shape=custom_shape,
    buffer_size=client_buffer_size, # Use the large client buffer size
    poll_interval=0.05 # Poll reasonably fast
)

# Register error callback for debugging
def on_error(error_msg):
    print(f"Connection Error: {error_msg}")
stream.register_error_callback(on_error)

# Connect to the server
stream.connect()
    
# Wait for connection to establish
print("Waiting for connection...")
time.sleep(2.0) # Give time for the connection handshake

if not stream.connected:
    print(f"Failed to connect to server: {stream.connection_error}")
    sys.exit(1) # Exit if connection failed

print(f"Connected. Polling frames for {poll_time_seconds} seconds...")
# Let the client's background polling thread run
time.sleep(poll_time_seconds)

print("Polling complete. Analyzing client buffer...")

#%%
# Get the contiguous buffer from the client's accumulated frames
# This will create an array from the min to max frame count *currently in the client buffer*
frames_array, counts_array, times_array = stream.get_contiguous_buffer()

# %%
if counts_array is None:
    print("No frames received in the client buffer.")
else:
    bad_frame_count = 0
    if frames_array is not None: # Should not be None if counts_array is not None
        # Iterate through the frames we expected based on counts_array
        for i in range(len(counts_array)):
            # Check if the corresponding frame in frames_array is all NaN
            # np.isnan().any() might be true for partial NaNs if data type allows
            # Check if the entire slice is NaN, indicating it was missing
            if np.all(np.isnan(frames_array[i])):
                bad_frame_count += 1
                if verbose: # Only print individual NaNs if verbose
                     print(f"Frame count {counts_array[i]} (index {i}) is missing (NaN)")
            
    total_frames_in_range = len(counts_array)
    actual_frames_received = stream.max_buffer_size - bad_frame_count # This isn't quite right
    
    # Calculate actual frames received by counting non-NaN frames
    actual_frames_received = total_frames_in_range - bad_frame_count

    print(f"\n--- Analysis ---")
    print(f"Client buffer min frame: {stream.buffer_min_frame}")
    print(f"Client buffer max frame: {stream.buffer_max_frame}")
    print(f"Total frame range spanned in client buffer: {total_frames_in_range} frames (from {counts_array[0]} to {counts_array[-1]})")
    print(f"Frames actually present in client buffer: {actual_frames_received}")
    print(f"Missing frames (gaps) within the spanned range: {bad_frame_count}")
    if total_frames_in_range > 0:
        loss_percentage = (bad_frame_count / total_frames_in_range) * 100
        print(f"Frame loss percentage: {loss_percentage:.2f}%")


 
# %%
