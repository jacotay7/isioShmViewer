import argparse
import time
import datetime
import numpy as np
import logging
import csv
import sys
import os
import matplotlib.pyplot as plt
from datetime import timedelta
import random

# Add the parent directory to sys.path to find isio_wrapper
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

try:
    from isio_wrapper import ImageStreamIO
except ImportError:
    print("Error: Could not import ImageStreamIO from isio_wrapper.py.")
    print(f"Ensure isio_wrapper.py is in the parent directory: {parent_dir} or in the Python path.")
    sys.exit(1)

# Setup logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def loss_function(data_array):
    """
    Placeholder loss function. Calculates the mean of the input array.
    Replace with actual metric calculation.

    Args:
        data_array (np.ndarray): Data read from the measurement SHM.

    Returns:
        float: Calculated loss metric.
    """
    if data_array is None or data_array.size == 0:
        return 0.0
    return np.mean(data_array)

def plot_latency_results(results, output_basename=None):
    """
    Create a time series plot of the latency measurement results.
    
    Args:
        results (list): List of dictionaries containing measurement results
        output_basename (str, optional): Base filename to save the plot
    """
    if not results:
        logger.warning("No results to plot")
        return
    
    try:
        # Convert string timestamps to datetime objects
        disturbance_time = datetime.datetime.fromisoformat(results[0]['disturbance_ts'])
        measurement_times = [datetime.datetime.fromisoformat(r['measurement_ts']) for r in results]
        loss_values = [r['loss'] for r in results]
        
        # Calculate time differences in milliseconds from disturbance
        time_diffs = [(t - disturbance_time).total_seconds() * 1000 for t in measurement_times]
        
        plt.figure(figsize=(10, 6))
        
        # Plot the loss values over time
        plt.plot(time_diffs, loss_values, 'o-', label='Measurement Loss')
        
        # Add a vertical line at the disturbance time (t=0)
        plt.axvline(x=0, color='r', linestyle='--', label='Disturbance Time')
        
        # Add labels and title
        plt.xlabel('Time Since Disturbance (ms)')
        plt.ylabel('Measurement Loss Value')
        plt.title('Latency Measurement Results')
        plt.grid(True, alpha=0.3)
        plt.legend(loc='best')
        
        # Add annotations for specific points
        for i, (t, loss) in enumerate(zip(time_diffs, loss_values)):
            plt.annotate(f"{i+1}", (t, loss), textcoords="offset points", 
                         xytext=(0,5), ha='center')
        
        # Format plot
        plt.tight_layout()
        
        # Save plot if output basename is provided
        if output_basename:
            plot_filename = f"{output_basename}_plot.png"
            plt.savefig(plot_filename)
            logger.info(f"Plot saved to: {plot_filename}")
        
        # Show plot
        plt.show()
        
    except Exception as e:
        logger.error(f"Error plotting results: {e}")

def main(args):
    """
    Main function to measure latency.
    """
    logger.info("Starting latency measurement...")
    logger.info(f"Disturbance SHM: {args.disturbance_shm}")
    logger.info(f"Measurement SHM: {args.measurement_shm}")
    logger.info(f"Command SHM: {args.command_shm}")
    logger.info(f"Disturbance Value: {args.disturbance_value}")
    logger.info(f"Frames to Capture (N): {args.num_frames}")
    logger.info(f"Output File: {args.output_file}")
    logger.info("Time variation: +/- 100ms")

    # --- Open SHM objects ---
    dist_shm = ImageStreamIO.open_shm(args.disturbance_shm)
    meas_shm = ImageStreamIO.open_shm(args.measurement_shm)
    cmd_shm = ImageStreamIO.open_shm(args.command_shm)

    if not dist_shm:
        logger.error(f"Failed to open disturbance SHM: {args.disturbance_shm}")
        sys.exit(1)
    if not meas_shm:
        logger.error(f"Failed to open measurement SHM: {args.measurement_shm}")
        sys.exit(1)
    if not cmd_shm:
        logger.error(f"Failed to open command SHM: {args.command_shm}")
        sys.exit(1)

    logger.info("Successfully opened all SHM objects.")

    # --- Prepare disturbance data ---
    try:
        dist_shape = ImageStreamIO.get_shape(dist_shm)
        disturbance_data = np.full(dist_shape, args.disturbance_value, dtype=np.float32)
        zero_data = np.zeros(dist_shape, dtype=np.float32)
        logger.info(f"Disturbance data shape: {dist_shape}")
    except Exception as e:
        logger.error(f"Failed to get shape or create data for disturbance SHM: {e}")
        sys.exit(1)

    # --- Initialize state ---
    results = []
    last_meas_cnt = -1
    last_cmd_cnt = -1
    measurement_frames_collected = 0
    monitoring_active = False
    disturbance_sent_ts = None

    # --- Read initial command counter ---
    # Read a few times to ensure we get a stable value if needed
    for _ in range(5):
        _, initial_cmd_cnt, _ = ImageStreamIO.read_shm(cmd_shm)
        if initial_cmd_cnt is not None:
            last_cmd_cnt = initial_cmd_cnt
            break
        time.sleep(0.01) # Small delay before retrying
        
    if last_cmd_cnt is None:
        logger.error("Failed to read initial command counter.")
        sys.exit(1)
    logger.info(f"Initial command SHM counter: {last_cmd_cnt}")
    
    # --- Send Disturbance with time variation ---
    # Generate random time variation between -100 and 100 ms
    time_variation_ms = random.uniform(-100, 100)
    logger.info(f"Applying time variation of {time_variation_ms:.2f} ms")
    
    # Sleep for the time variation if positive, otherwise proceed immediately
    if time_variation_ms > 0:
        time.sleep(time_variation_ms / 1000.0)
    
    disturbance_sent_ts = datetime.datetime.now()
    ret = ImageStreamIO.write_shm(dist_shm, disturbance_data)
    if ret != 0:
        logger.error("Failed to write disturbance to SHM.")
        sys.exit(1)
    logger.info(f"Disturbance sent at: {disturbance_sent_ts.isoformat()} (with {time_variation_ms:.2f} ms variation)")
    monitoring_active = True
    
    # --- Monitoring Loop ---
    try:
        while monitoring_active:
            # --- Monitor Measurement SHM ---
            meas_data, meas_cnt, meas_ts = ImageStreamIO.read_shm(meas_shm)
            
            # Check if it's a new frame and if we are still collecting
            if meas_cnt is not None and meas_cnt != last_meas_cnt and measurement_frames_collected < args.num_frames:
                last_meas_cnt = meas_cnt
                # Ensure the measurement happened at or after the disturbance
                if meas_ts >= disturbance_sent_ts:
                    loss = loss_function(meas_data)
                    results.append({
                        "disturbance_ts": disturbance_sent_ts.isoformat(),
                        "measurement_ts": meas_ts.isoformat(),
                        "measurement_cnt": meas_cnt,
                        "time_variation_ms": time_variation_ms,
                        "loss": loss
                    })
                    measurement_frames_collected += 1
                    logger.debug(f"Collected measurement frame {measurement_frames_collected}/{args.num_frames} (cnt: {meas_cnt}, ts: {meas_ts.isoformat()})")

            # --- Monitor Command SHM ---
            _, cmd_cnt, cmd_ts = ImageStreamIO.read_shm(cmd_shm)

            if cmd_cnt is not None and cmd_cnt != last_cmd_cnt:
                # Check if the counter actually increased (handles potential resets)
                if cmd_cnt > last_cmd_cnt:
                    logger.info(f"Command SHM counter changed: {last_cmd_cnt} -> {cmd_cnt} at {cmd_ts.isoformat()}")

                    # Reset disturbance SHM
                    ret = ImageStreamIO.write_shm(dist_shm, zero_data)
                    if ret != 0:
                        logger.warning("Failed to write zero data to disturbance SHM.")
                    else:
                        logger.info("Disturbance SHM reset to zero.")

                    monitoring_active = False  # Stop monitoring after command change and reset
                else:
                    # Counter changed but didn't increase (e.g., reset externally)
                    # Update our baseline counter to the new value
                    logger.warning(f"Command SHM counter changed but did not increase: {last_cmd_cnt} -> {cmd_cnt}. Updating baseline.")
                    last_cmd_cnt = cmd_cnt

            # Check if we've collected enough frames
            if measurement_frames_collected >= args.num_frames:
                logger.info(f"Collected {measurement_frames_collected} frames. Stopping measurement.")

                # Reset disturbance SHM
                ret = ImageStreamIO.write_shm(dist_shm, zero_data)
                if ret != 0:
                    logger.warning("Failed to write zero data to disturbance SHM.")
                else:
                    logger.info("Disturbance SHM reset to zero.")

                monitoring_active = False

            # Prevent busy-waiting
            time.sleep(0.001)  # Sleep for 1ms




