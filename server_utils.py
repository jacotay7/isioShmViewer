import socket
import logging
import struct
import time
import threading
from isio_wrapper import ImageStreamIO

# Setup logger
logger = logging.getLogger(__name__)

class BandwidthManager:
    """
    Manages bandwidth usage for data transmission.
    """
    def __init__(self, max_bandwidth_bps):
        self.max_bandwidth = max_bandwidth_bps
        self.start_time = time.time()
        self.bytes_sent = 0

    def update(self, bytes_to_send):
        """
        Update bandwidth usage and return required sleep time.
        
        Args:
            bytes_to_send (int): Number of bytes to be sent
            
        Returns:
            float: Time to sleep in seconds to maintain bandwidth limit
        """
        self.bytes_sent += bytes_to_send
        elapsed = time.time() - self.start_time
        target_time = self.bytes_sent / self.max_bandwidth
        return max(0, target_time - elapsed)

    def reset(self):
        """Reset the bandwidth manager counters."""
        self.start_time = time.time()
        self.bytes_sent = 0

class ServerBase:
    """
    Base class for shared memory streaming servers.
    """
    def __init__(self, host='127.0.0.1', port=5123, max_clients=None, bw_limit_mbps=10.0):
        self.host = host
        self.port = port
        self.max_clients = max_clients
        self.bw_limit_bps = bw_limit_mbps * 1024 * 1024
        self.running = False
        self.socket = None
        self.client_threads = []
        
    def start(self):
        """
        Start the server to accept client connections.
        """
        self.running = True
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.socket.bind((self.host, self.port))
            self.socket.listen()
            
            logger.info(f"Server listening on {self.host}:{self.port}")
            if self.max_clients:
                logger.info(f"Maximum clients allowed: {self.max_clients}")
            else:
                logger.info("No client limit set.")
            logger.info(f"Default bandwidth limit per client: {self.bw_limit_bps / (1024*1024)} MB/s")

            while self.running:
                try:
                    # Clean up terminated threads
                    self.client_threads = [t for t in client_threads if t.is_alive()]
                    active_client_count = len(self.client_threads)
                    logger.debug(f"Active clients: {active_client_count}")

                    if self.max_clients is not None and active_client_count >= self.max_clients:
                        logger.warning(f"Max client limit ({self.max_clients}) reached. Waiting for a slot...")
                        time.sleep(1)
                        continue

                    conn, addr = self.socket.accept()
                    logger.info(f"New connection from {addr}")
                    client_thread = threading.Thread(
                        target=self.handle_client,
                        args=(conn, addr, self.bw_limit_bps),
                        daemon=True
                    )
                    client_thread.start()
                    self.client_threads.append(client_thread)

                except KeyboardInterrupt:
                    logger.info("Shutdown signal received.")
                    self.running = False
                    break
                except Exception as e:
                    logger.error(f"Error accepting connection: {str(e)}", exc_info=True)
                    time.sleep(1)
                    
        except Exception as e:
            logger.critical(f"Server error: {str(e)}", exc_info=True)
        finally:
            self.shutdown()

    def shutdown(self):
        """
        Shutdown the server and clean up resources.
        """
        self.running = False
        logger.info("Shutting down server.")
        
        if self.socket:
            try:
                self.socket.close()
            except Exception as e:
                logger.error(f"Error closing server socket: {str(e)}")
    
    def handle_client(self, conn, addr, bw_limit_bps):
        """
        Abstract method for handling client connections.
        Must be implemented by derived classes.
        """
        raise NotImplementedError("Subclasses must implement handle_client()")

def recv_all(sock, n):
    """
    Helper function to receive exactly n bytes from a socket.
    
    Args:
        sock (socket): Socket to receive from
        n (int): Number of bytes to receive
        
    Returns:
        bytes or None: Received data or None if connection closed
    """
    data = b''
    while len(data) < n:
        packet = sock.recv(n - len(data))
        if not packet:
            return None
        data += packet
    return data

def pin_to_cores(cores):
    """
    Pin the current process to specific CPU cores.

    Args:
        cores (list): A list of core numbers to pin the process to
    """
    try:
        import psutil
        p = psutil.Process()
        p.cpu_affinity(cores)
        logger.info(f"Process pinned to CPU cores: {cores}")
    except ImportError:
        logger.warning("psutil module not available. CPU pinning not supported.")
    except AttributeError:
        logger.warning("CPU affinity setting not available on this system/psutil version.")
    except Exception as e:
        logger.error(f"Failed to pin process to cores {cores}: {str(e)}")