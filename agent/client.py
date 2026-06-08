import socket
import json
import time
import psutil
import threading
import logging
from datetime import datetime
logging.basicConfig(level=logging.INFO, format='%(levelname)s\t%(message)s')
logger = logging.getLogger(__name__)
class NetworkAgent:
    def __init__(self, server_host='localhost', server_port=8888, agent_name='localhost', api_key=None):
        self.server_host = server_host
        self.server_port = server_port
        self.agent_name = agent_name
        self.api_key = api_key
        self.is_running = False
        self.thread = None
        self.last_stats = None
    def _collect_data(self):
        try:
            stats = psutil.net_io_counters()
            connections = psutil.net_connections()
            interfaces = psutil.net_if_stats()
            bytes_sent = 0
            bytes_recv = 0
            packets_sent = 0
            packets_recv = 0
            if self.last_stats:
                bytes_sent = stats.bytes_sent - self.last_stats.bytes_sent
                bytes_recv = stats.bytes_recv - self.last_stats.bytes_recv
                packets_sent = stats.packets_sent - self.last_stats.packets_sent
                packets_recv = stats.packets_recv - self.last_stats.packets_recv
            self.last_stats = stats
            active_connections = len([c for c in connections if c.status == 'ESTABLISHED'])
            active_interfaces = len([i for i, s in interfaces.items() if s.isup])
            features = self._generate_features(bytes_sent, bytes_recv, packets_sent, packets_recv)
            return {
                'agent_name': self.agent_name,
                'timestamp': datetime.now().isoformat(),
                'bytes_sent': bytes_sent,
                'bytes_recv': bytes_recv,
                'packets_sent': packets_sent,
                'packets_recv': packets_recv,
                'active_connections': active_connections,
                'active_interfaces': active_interfaces,
                'features': features
            }
        except Exception as e:
            logger.error(f" : {e}")
            return None
    def _generate_features(self, bytes_sent, bytes_recv, packets_sent, packets_recv):
        total_bytes = bytes_sent + bytes_recv
        total_packets = packets_sent + packets_recv
        features = [
            min(total_bytes / 1e6, 1.0),
            min(total_packets / 1e3, 1.0),
            min(bytes_sent / 1e6, 1.0),
            min(bytes_recv / 1e6, 1.0),
            min(packets_sent / 1e3, 1.0),
            min(packets_recv / 1e3, 1.0),
            min(bytes_sent / (bytes_recv + 1e-9), 1.0),
            min(total_bytes / (total_packets + 1e-9) / 1500, 1.0)
        ]
        return features
    def send_data(self):
        data = self._collect_data()
        if not data:
            return False
        try:
            import requests
            url = f"http://{self.server_host}:{self.server_port}/api/agent/data"
            headers = {}
            if self.api_key:
                headers['X-API-Key'] = self.api_key
            response = requests.post(url, json=data, headers=headers, timeout=5)
            if response.status_code == 200:
                return True
            else:
                logger.error(f" : {response.status_code} - {response.text}")
                return False
        except Exception as e:
            logger.error(f" : {e}")
            return False
    def _run(self):
        while self.is_running:
            self.send_data()
            time.sleep(1)
    def start(self):
        if not self.is_running:
            self.is_running = True
            self.thread = threading.Thread(target=self._run, daemon=True)
            self.thread.start()
            logger.info(f"  {self.agent_name}  {self.server_host}:{self.server_port}")
    def stop(self):
        self.is_running = False
        if self.thread:
            self.thread.join(timeout=2)
        logger.info(f"  {self.agent_name} ")
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Network Security Agent')
    parser.add_argument('--host', default='localhost', help='Server host')
    parser.add_argument('--port', type=int, default=8000, help='Server port')
    parser.add_argument('--name', default='agent-1', help='Agent name')
    parser.add_argument('--api-key', default=None, help='API key for authentication')
    args = parser.parse_args()
    agent = NetworkAgent(server_host=args.host, server_port=args.port, agent_name=args.name, api_key=args.api_key)
    agent.start()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        agent.stop()
