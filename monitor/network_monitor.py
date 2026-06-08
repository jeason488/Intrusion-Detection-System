import psutil
import socket
import time
import threading
import queue
import logging
import numpy as np
from datetime import datetime
logging.basicConfig(level=logging.INFO, format='%(levelname)s\t%(message)s')
logger = logging.getLogger(__name__)
class NetworkMonitor:
    def __init__(self):
        self.is_monitoring = False
        self.monitor_thread = None
        self.data_queue = queue.Queue()
        self.last_stats = None
        self.detection_results = []
    def start_monitoring(self):
        if not self.is_monitoring:
            self.is_monitoring = True
            self.monitor_thread = threading.Thread(target=self._monitor_network, daemon=True)
            self.monitor_thread.start()
            logger.info(" ")
    def stop_monitoring(self):
        if self.is_monitoring:
            self.is_monitoring = False
            if self.monitor_thread:
                self.monitor_thread.join(timeout=2.0)
            logger.info(" ")
    def _monitor_network(self):
        while self.is_monitoring:
            try:
                stats = psutil.net_io_counters()
                if self.last_stats:
                    bytes_sent = stats.bytes_sent - self.last_stats.bytes_sent
                    bytes_recv = stats.bytes_recv - self.last_stats.bytes_recv
                    packets_sent = stats.packets_sent - self.last_stats.packets_sent
                    packets_recv = stats.packets_recv - self.last_stats.packets_recv
                    err_in = stats.errin - self.last_stats.errin
                    err_out = stats.errout - self.last_stats.errout
                    drop_in = stats.dropin - self.last_stats.dropin
                    drop_out = stats.dropout - self.last_stats.dropout
                    features = self._generate_features(
                        bytes_sent, bytes_recv, packets_sent, packets_recv,
                        err_in, err_out, drop_in, drop_out
                    )
                    connections = psutil.net_connections()
                    active_connections = len([c for c in connections if c.status == 'ESTABLISHED'])
                    interfaces = psutil.net_if_stats()
                    active_interfaces = len([i for i, s in interfaces.items() if s.isup])
                    features.extend([active_connections, active_interfaces])
                    if len(features) < 10:
                        features.extend([0] * (10 - len(features)))
                    elif len(features) > 10:
                        features = features[:10]
                    timestamp = datetime.now().isoformat()
                    self.data_queue.put({
                        'timestamp': timestamp,
                        'features': features,
                        'raw_data': {
                            'bytes_sent': bytes_sent,
                            'bytes_recv': bytes_recv,
                            'packets_sent': packets_sent,
                            'packets_recv': packets_recv,
                            'err_in': err_in,
                            'err_out': err_out,
                            'drop_in': drop_in,
                            'drop_out': drop_out,
                            'active_connections': active_connections,
                            'active_interfaces': active_interfaces
                        }
                    })
                self.last_stats = stats
                time.sleep(0.5)
            except Exception as e:
                logger.error(f" : {e}")
                time.sleep(1)
    def _generate_features(self, bytes_sent, bytes_recv, packets_sent, packets_recv,
                          err_in, err_out, drop_in, drop_out):
        total_bytes = bytes_sent + bytes_recv
        total_packets = packets_sent + packets_recv
        error_rate_in = (err_in + drop_in) / (packets_recv + 1e-9)
        error_rate_out = (err_out + drop_out) / (packets_sent + 1e-9)
        send_recv_ratio = bytes_sent / (bytes_recv + 1e-9)
        packet_size_avg = total_bytes / (total_packets + 1e-9)
        features = [
            min(total_bytes / 1e6, 1.0),
            min(total_packets / 1e3, 1.0),
            min(error_rate_in, 1.0),
            min(error_rate_out, 1.0),
            min(send_recv_ratio, 1.0),
            min(packet_size_avg / 1500, 1.0),
            min(bytes_sent / 1e6, 1.0),
            min(bytes_recv / 1e6, 1.0)
        ]
        return features
    def get_latest_data(self):
        data_list = []
        while not self.data_queue.empty():
            data_list.append(self.data_queue.get())
        return data_list
    def add_detection_result(self, result):
        self.detection_results.append(result)
        if len(self.detection_results) > 100:
            self.detection_results = self.detection_results[-100:]
    def get_detection_history(self):
        return self.detection_results
network_monitor = NetworkMonitor()
