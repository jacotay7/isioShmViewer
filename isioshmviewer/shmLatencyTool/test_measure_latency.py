import argparse
import time
import datetime
import numpy as np
import logging
import sys
import os
import threading
import signal
import math

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

# Global flag to signal threads to stop
running = True

# Global variables to track disturbance events
latest_disturbance_time = None
latest_disturbance_data = None
disturbance_processed = False

def signal_handler(sig, frame):
    global running
    logger.info("Signal received, stopping simulation...")
    running = False

def gaussian_response(t, mean_ms=200, std_ms=30):
    """
    Calculate Gaussian response for a given time difference.
    
    Args:
        t (float): Time in milliseconds since disturbance
        mean_ms (float): Center of the Gaussian in milliseconds
        std_ms (float): Standard deviation of the Gaussian in milliseconds
        
    Returns:
        float: Gaussian response value between 0 and 1
    """
    return math.exp(-((t - mean_ms) ** 2) / (2 * std_ms ** 2))

def disturbance_listener(dist_shm, cmd_shm):
    """
    Listens to the disturbance SHM and sets global variables when disturbance detected.
    """
    global latest_disturbance_time, latest_disturbance_data, disturbance_processed
    
    logger.info("Disturbance listener thread started.")
    last_dist_cnt = -1
    while running:
        try:
            dist_data, dist_cnt, dist_ts = ImageStreamIO.read_shm(dist_shm)
            
            if dist_cnt is not None and dist_cnt != last_dist_cnt:
                logger.info(f"Disturbance detected (cnt: {dist_cnt}, ts: {dist_ts.isoformat()})")
                last_dist_cnt = dist_cnt
                
                # Record the disturbance time and data for the main loop to use
                if np.any(dist_data):  # Check if there's any non-zero value
                    latest_disturbance_time = time.perf_counter()
                    latest_disturbance_data = dist_data.copy()
                    disturbance_processed = False
                    logger.info(f"New disturbance recorded at {latest_disturbance_time}, mean value: {np.mean(latest_disturbance_data)}")

                # Still update command as before for compatibility
                cmd_data, cmd_cnt, cmd_ts = ImageStreamIO.read_shm(cmd_shm)
                if cmd_data is not None and dist_data is not None:
                    if cmd_data.shape == dist_data.shape:
                        new_cmd_data = cmd_data + dist_data
                        logger.debug(f"Adding disturbance to command. New value mean: {np.mean(new_cmd_data)}")
                        ret = ImageStreamIO.write_shm(cmd_shm, new_cmd_data)
                        if ret != 0:
                             logger.warning("Listener failed to write updated command data.")
                    else:
                        logger.warning(f"Shape mismatch between command ({cmd_data.shape}) and disturbance ({dist_data.shape}).")
            
            # Sleep briefly to avoid busy-waiting
            time.sleep(0.001) 

        except Exception as e:
            logger.error(f"Error in disturbance listener: {e}")
            time.sleep(0.1) # Avoid spamming errors

    logger.info("Disturbance listener thread stopped.")


def main(args):
    """
    Main simulation loop.
    """
    global running, latest_disturbance_time, latest_disturbance_data, disturbance_processed
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    logger.info("Starting SHM simulation...")
    shape = tuple(map(int, args.shape.split(',')))
    logger.info(f"Using shape: {shape}")
    target_period = 1.0 / args.rate # Target time per frame in seconds
    logger.info(f"Target rate: {args.rate} Hz (Period: {target_period:.4f} s)")

    # --- Create SHM objects ---
    dist_shm = ImageStreamIO.create_shm(args.disturbance_shm, shape, np.float32)
    meas_shm = ImageStreamIO.create_shm(args.measurement_shm, shape, np.float32)
    cmd_shm = ImageStreamIO.create_shm(args.command_shm, shape, np.float32)

    if not dist_shm or not meas_shm or not cmd_shm:
        logger.error("Failed to create one or more SHM objects. Exiting.")
        sys.exit(1)

    # --- Initialize SHM data ---
    initial_cmd_data = np.random.rand(*shape).astype(np.float32) * 10 # Start with some random data
    ImageStreamIO.write_shm(cmd_shm, initial_cmd_data)
    ImageStreamIO.write_shm(meas_shm, np.zeros(shape, dtype=np.float32))
    ImageStreamIO.write_shm(dist_shm, np.zeros(shape, dtype=np.float32))
    logger.info("SHM objects created and initialized.")

    # --- Start disturbance listener thread ---
    listener_thread = threading.Thread(target=disturbance_listener, args=(dist_shm, cmd_shm), daemon=True)
    listener_thread.start()

    # --- Main Simulation Loop ---
    logger.info("Starting main simulation loop...")
    frame_count = 0
    last_loop_time = time.perf_counter()

    try:
        while running:
            loop_start_time = time.perf_counter()

            # 1. Read current command data
            cmd_data, _, _ = ImageStreamIO.read_shm(cmd_shm)
            if cmd_data is None:
                logger.warning("Failed to read command data in main loop. Skipping frame.")
                time.sleep(target_period / 2) # Wait before retrying
                continue
                
            # 2. Calculate 80% and write back to command
            new_cmd_data = (cmd_data * 0.8).astype(np.float32)
            ret_cmd = ImageStreamIO.write_shm(cmd_shm, new_cmd_data)
            if ret_cmd != 0:
                 logger.warning("Main loop failed to write command data.")

            # 3. Check for disturbance and apply Gaussian response to measurement
            meas_data = np.zeros_like(new_cmd_data)
            
            if latest_disturbance_time is not None:
                # Calculate time since disturbance in milliseconds
                elapsed_time_ms = (time.perf_counter() - latest_disturbance_time) * 1000.0
                
                # Calculate Gaussian response (peaks at 200ms, std dev of 30ms)
                response = gaussian_response(elapsed_time_ms)
                
                # Apply the response to the measurement
                if latest_disturbance_data is not None and not disturbance_processed:
                    amplitude = np.mean(latest_disturbance_data) * 10  # Using disturbance mean as amplitude scale
                    meas_data = (new_cmd_data + (amplitude * response)).astype(np.float32)
                    
                    logger.debug(f"Applied Gaussian response: {response:.4f} at {elapsed_time_ms:.1f}ms")
                    
                    # If we're past the Gaussian peak (200ms) by 3 standard deviations, mark as processed
                    if elapsed_time_ms > 200 + (3 * 30):
                        logger.info(f"Disturbance response complete after {elapsed_time_ms:.1f}ms")
                        disturbance_processed = True
                else:
                    meas_data = new_cmd_data.copy()
            else:
                meas_data = new_cmd_data.copy()
                
            # 4. Write the measurement data
            ret_meas = ImageStreamIO.write_shm(meas_shm, meas_data)
            if ret_meas != 0:
                 logger.warning("Main loop failed to write measurement data.")

            frame_count += 1

            # --- Timing control ---
            loop_end_time = time.perf_counter()
            elapsed_time = loop_end_time - loop_start_time
            sleep_time = target_period - elapsed_time
            
            if sleep_time > 0:
                time.sleep(sleep_time)
            else:
                logger.warning(f"Loop iteration {frame_count} took too long: {elapsed_time:.4f}s > {target_period:.4f}s")
            
            last_loop_time = time.perf_counter()


    except Exception as e:
         logger.error(f"Error in main simulation loop: {e}", exc_info=True)
    finally:
        logger.info("Stopping simulation loop...")
        running = False # Ensure flag is set for the thread
        if listener_thread.is_alive():
             listener_thread.join(timeout=1.0) # Wait briefly for thread
        logger.info("Simulation finished.")
        
        print("\n--- Instructions for running measure_latency.py ---")
        print("1. Keep this test script running.")
        print("2. Open a new terminal.")
        print("3. Navigate to the directory containing measure_latency.py:")
        print(f"   cd {os.path.dirname(os.path.abspath(__file__))}")
        print("4. Run measure_latency.py, providing the SHM names used by this script:")
        cmd = (f"   python measure_latency.py {args.disturbance_shm} {args.measurement_shm} {args.command_shm} "
               f"-n 20 -o latency_results.csv --debug")
        print(cmd)
        print("   (Adjust -n, -o, --disturbance_value as needed)")
        print("5. The measure_latency script will send a disturbance, collect data, plot results, and exit.")
        print("6. You can then stop this test script (Ctrl+C).")
        print("----------------------------------------------------\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simulate SHM environment for latency testing.")
    
    parser.add_argument("--disturbance-shm", default="dist_shm_test", help="Name for the disturbance SHM.")
    parser.add_argument("--measurement-shm", default="meas_shm_test", help="Name for the measurement SHM.")
    parser.add_argument("--command-shm", default="cmd_shm_test", help="Name for the command SHM.")
    parser.add_argument("--shape", default="64,64", help="Shape of the SHM data (e.g., 'height,width').")
    parser.add_argument("--rate", type=float, default=100.0, help="Target simulation rate in Hz.")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging.")

    parsed_args = parser.parse_args()

    if parsed_args.debug:
        logger.setLevel(logging.DEBUG)
        logging.getLogger().setLevel(logging.DEBUG)

    main(parsed_args)
