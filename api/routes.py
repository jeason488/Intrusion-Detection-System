from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
import torch
import numpy as np
import sqlite3
import json
import logging
from datetime import datetime
from collections import deque
import os
from models.dhpen_network import DHPENNetwork
from llm.model_interface import LLMAuxiliary as LLMInterface
from monitor.network_monitor import network_monitor
from drift.attribution import DriftAttribution
from drift.adaptation_controller import AdaptationController
logging.basicConfig(level=logging.INFO, format='%(levelname)s\t%(message)s')
logger = logging.getLogger(__name__)
def init_database():
    conn = sqlite3.connect('detection_results.db')
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS detections (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            is_anomaly INTEGER,
            confidence REAL,
            anomaly_score REAL,
            attack_type TEXT,
            drift_detected INTEGER,
            features TEXT
        )
    ''')
    conn.commit()
    conn.close()
init_database()
class DetectionRequest(BaseModel):
    features: list = Field(..., description="")
    use_llm: bool = Field(False, description="")
class DetectionResponse(BaseModel):
    timestamp: str
    is_anomaly: bool
    confidence: float
    anomaly_score: float
    attack_type: str
    drift_detected: bool
    drift_score: float
    llm_explanation: str = ""
    features: list
class RealTimeMonitorResponse(BaseModel):
    timestamp: str
    network_data: dict
    detection_result: DetectionResponse = None
detection_history = []
model = None
llm_interface = None
registered_agents = {}
agent_data_cache = {}
def init_model():
    global model, llm_interface
    try:
        import yaml
        with open('config/config.yaml', 'r', encoding='utf-8') as f:
            yaml_config = yaml.safe_load(f)
        config = {
            'input_dim': yaml_config.get('data', {}).get('input_dim', 59),
            'hidden_dim': 256,
            'prototype_dim': yaml_config.get('models', {}).get('prototype', {}).get('dim', 128),
            'num_heads': 4, 'num_layers': 2,
            'n_way': yaml_config.get('models', {}).get('meta_learning', {}).get('n_way', 5),
            'k_shot': yaml_config.get('models', {}).get('meta_learning', {}).get('k_shot', 5),
            'n_query': yaml_config.get('models', {}).get('meta_learning', {}).get('n_query', 15),
            'inner_lr': yaml_config.get('models', {}).get('meta_learning', {}).get('inner_lr', 0.01),
            'meta_lr': yaml_config.get('models', {}).get('meta_learning', {}).get('meta_lr', 1e-4),
            'adaptation_steps': yaml_config.get('models', {}).get('meta_learning', {}).get('adaptation_steps', 5),
            'alpha_fusion': yaml_config.get('models', {}).get('prototype', {}).get('alpha_fusion', 0.7),
            'beta_update': yaml_config.get('models', {}).get('prototype', {}).get('beta_update', 0.9),
            'temperature': 18.0, 'device': 'cpu', 'feature_dim': 59,
            'use_complexity_estimator': True, 'use_hierarchy_constructor': True,
            'use_evolution_engine': True, 'use_global_memory': True,
            'checkpoint_dir': './checkpoints'
        }
        model = DHPENNetwork(config)
        checkpoint_path = './checkpoints/model_checkpoint.pth'
        if os.path.exists(checkpoint_path):
            try:
                checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
                model.load_state_dict(checkpoint, strict=False)
                logger.info("")
            except Exception as e:
                logger.warning(f": {e}")
        model.eval()
        llm_config = {'model_name': 'bert-base-uncased', 'max_seq_length': 128, 'device': 'cpu'}
        llm_yaml_config = yaml_config.get('llm', {}) if yaml_config else {}
        if llm_yaml_config.get('use_deepseek', False):
            llm_config['use_deepseek'] = True
            llm_config['deepseek_api_key'] = llm_yaml_config.get('deepseek_api_key', '')
            llm_config['deepseek_api_url'] = llm_yaml_config.get('deepseek_api_url', 'https://api.deepseek.com/v1/chat/completions')
            llm_config['deepseek_model'] = llm_yaml_config.get('deepseek_model', 'deepseek-chat')
        llm_interface = LLMInterface(llm_config)
        logger.info("")
    except Exception as e:
        logger.error(f": {e}")
init_model()
router = APIRouter()
def detect_drift(current_features, historical_features, threshold=0.3):
    if len(historical_features) < 5:
        return False, 0.0
    current_mean, current_std = np.mean(current_features), np.std(current_features)
    historical_mean, historical_std = np.mean(historical_features), np.std(historical_features)
    drift_score = (abs(current_mean - historical_mean) + abs(current_std - historical_std)) / 2
    return drift_score > threshold, drift_score
def get_enhanced_llm_explanation(features, is_anomaly, confidence, attack_type, drift_detected, drift_score):
    if not llm_interface:
        return ""
    try:
        status = "" if is_anomaly else ""
        drift_status = "" if drift_detected else ""
        explanation = f"{status}{confidence:.2f}{attack_type}{drift_status}"
        result = llm_interface.generate_detection_explanation(features, 1 if is_anomaly else 0, confidence)
        if result and 'suggestions' in result:
            explanation += "\n" + ", ".join(result['suggestions'])
        return explanation
    except Exception as e:
        logger.error(f": {e}")
        return ""
def save_detection_result(result):
    try:
        conn = sqlite3.connect('detection_results.db')
        cursor = conn.cursor()
        cursor.execute('INSERT INTO detection_results (timestamp, features, is_anomaly, confidence, anomaly_score, '
                       'attack_type, drift_detected, drift_score, llm_explanation, raw_result) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)',
                       (result['timestamp'], json.dumps(result['features']),
                        1 if result['is_anomaly'] else 0, result['confidence'], result['anomaly_score'],
                        result['attack_type'], 1 if result['drift_detected'] else 0,
                        result['drift_score'], result['llm_explanation'], json.dumps(result)))
        conn.commit()
        conn.close()
    except Exception as e:
        logger.error(f": {e}")
ATTACK_TYPES = {
    'BENIGN': '',
    'DDoS': 'DDoS',
    'PortScan': '',
    'FTP-Patator': 'FTP',
    'SSH-Patator': 'SSH',
    'DoS slowloris': 'DoS',
    'DoS Slowhttptest': 'HTTP',
    'DoS Hulk': 'Hulk DoS',
    'DoS GoldenEye': 'GoldenEye DoS',
    'Heartbleed': '',
    'Web Attack': 'Web',
    'XSS': 'XSS',
    'Sql Injection': 'SQL',
    'Infiltration': '',
    'Bot': '',
    'DDoS attacks': 'DDoS',
    'Network Scan': '',
    'Data Exfiltration': '',
    'Unknown': ''
}
def classify_attack_type(features, is_anomaly, confidence):
    if not is_anomaly:
        return ''
    if len(features) >= 6:
        total_bytes = features[0]
        total_packets = features[1]
        bytes_sent = features[2]
        bytes_recv = features[3]
        if total_packets > 0.8 or total_bytes > 0.8:
            return 'DDoS'
        if total_packets > 0.5 and total_bytes < 0.3:
            return ''
        if bytes_sent > 0.7 and bytes_sent > bytes_recv * 2:
            return ''
        if total_bytes > 0.5 and total_bytes < 0.8:
            return 'DoS'
        if total_packets > 0.3 and total_packets < 0.6:
            return ''
    if confidence > 0.9:
        if features[4] > 0.7:
            return ''
        elif features[5] > 0.7:
            return ''
    return ''
def expand_to_59d(features):
    base_features = list(features[:8])
    while len(base_features) < 8:
        base_features.append(0.0)
    expanded = []
    expanded.extend(base_features)
    total_bytes = base_features[0]
    total_packets = base_features[1]
    bytes_sent = base_features[2]
    bytes_recv = base_features[3]
    expanded.append(total_bytes * total_packets)
    expanded.append(bytes_sent / (bytes_recv + 1e-9))
    expanded.append(total_packets / (total_bytes + 1e-9))
    expanded.append(base_features[4] / (base_features[5] + 1e-9))
    expanded.append(base_features[6] * base_features[7])
    while len(expanded) < 59:
        expanded.append(float(torch.randn(1).item() * 0.1))
    return expanded[:59]
@router.post("/api/detect", response_model=DetectionResponse)
async def detect_anomaly(request: DetectionRequest):
    global detection_history, model
    try:
        features = request.features
        if len(features) < 8:
            features = list(features) + [0.0] * (8 - len(features))
        elif len(features) > 8:
            features = features[:8]
        original_features = features.copy()
        features_59d = expand_to_59d(features)
        features_tensor = torch.tensor(features_59d, dtype=torch.float32).unsqueeze(0)
        is_anomaly = False
        confidence = 0.5
        anomaly_score = 0.5
        if model is not None:
            try:
                with torch.no_grad():
                    support_x = torch.randn(10, 59) * 0.3
                    support_y = torch.tensor([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
                    result = model.meta_test_step(support_x, support_y, features_tensor)
                    if result is not None:
                        predictions = result.get('predictions', torch.tensor([0]))
                        is_anomaly = predictions[0].item() == 1
                        if 'confidence' in result:
                            confidence = float(result['confidence'].get('mean', 0.5))
                        else:
                            logits = result.get('logits', torch.tensor([[0.5, 0.5]]))
                            probs = torch.softmax(logits, dim=1)
                            confidence = float(probs[0, predictions[0]])
                        anomaly_score = float(1.0 - confidence)
            except Exception as e:
                logger.warning(f" : {e}")
                is_anomaly = original_features[0] > 0.5 or original_features[1] > 0.5
                confidence = min(0.95, original_features[0] + original_features[1])
                anomaly_score = confidence
        else:
            is_anomaly = original_features[0] > 0.5 or original_features[1] > 0.5
            confidence = min(0.95, original_features[0] + original_features[1])
            anomaly_score = confidence
        attack_type = classify_attack_type(original_features, is_anomaly, confidence)
        attack_types = ["", "DDoS", "", "", "", ""]
        if not is_anomaly:
            attack_type = ""
        else:
            if features[0] > 0.8 and features[1] > 0.8:
                attack_type = "DDoS"
            elif features[2] > 0.5 or features[3] > 0.5:
                attack_type = ""
            elif features[4] > 0.8:
                attack_type = ""
            elif len(features) > 8 and features[8] > 0.5:
                attack_type = ""
            else:
                attack_type = ""
        historical_features = [hist['features'] for hist in detection_history[-20:]]
        drift_detected, drift_score = detect_drift(features, historical_features)
        llm_explanation = ""
        if request.use_llm:
            llm_explanation = get_enhanced_llm_explanation(
                features, is_anomaly, confidence, attack_type, drift_detected, drift_score
            )
        timestamp = datetime.now().isoformat()
        detection_result = {
            'timestamp': timestamp,
            'features': features,
            'is_anomaly': is_anomaly,
            'confidence': confidence,
            'anomaly_score': anomaly_score,
            'attack_type': attack_type,
            'drift_detected': drift_detected,
            'drift_score': drift_score,
            'llm_explanation': llm_explanation
        }
        detection_history.append(detection_result)
        if len(detection_history) > 100:
            detection_history = detection_history[-100:]
        save_detection_result(detection_result)
        network_monitor.add_detection_result(detection_result)
        return DetectionResponse(**detection_result)
    except Exception as e:
        logger.error(f" : {e}")
        raise HTTPException(status_code=500, detail=f": {str(e)}")
@router.get("/api/monitor/realtime", response_model=list[RealTimeMonitorResponse])
async def get_realtime_data():
    try:
        network_data_list = network_monitor.get_latest_data()
        response_data = []
        for data in network_data_list:
            features = data['features']
            detection_request = DetectionRequest(features=features, use_llm=False)
            detection_response = await detect_anomaly(detection_request)
            monitor_response = RealTimeMonitorResponse(
                timestamp=data['timestamp'],
                network_data=data['raw_data'],
                detection_result=detection_response
            )
            response_data.append(monitor_response)
        if not response_data and len(detection_history) > 0:
            recent_results = detection_history[-10:]
            for idx, result in enumerate(recent_results):
                features = result.get('features', [])
                bytes_sent = features[2] * 100000 if len(features) > 2 else (idx + 1) * 100000
                bytes_recv = features[3] * 100000 if len(features) > 3 else (idx + 1) * 80000
                response_data.append({
                    'timestamp': result.get('timestamp', ''),
                    'network_data': {
                        'bytes_sent': bytes_sent,
                        'bytes_recv': bytes_recv,
                        'packets_sent': (idx + 1) * 5,
                        'packets_recv': (idx + 1) * 4
                    },
                    'detection_result': {
                        'is_anomaly': result.get('is_anomaly', False),
                        'confidence': result.get('confidence', 0.0),
                        'attack_type': result.get('attack_type', ''),
                        'anomaly_score': result.get('anomaly_score', 0.0),
                        'drift_detected': result.get('drift_detected', False),
                        'drift_score': result.get('drift_score', 0.0),
                        'llm_explanation': result.get('llm_explanation', ''),
                        'features': features
                    }
                })
        return response_data
    except Exception as e:
        logger.error(f" : {e}")
        raise HTTPException(status_code=500, detail=f": {str(e)}")
@router.get("/api/history")
async def get_history(limit: int = 100):
    try:
        conn = sqlite3.connect('detection_results.db')
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM detection_results ORDER BY timestamp DESC LIMIT ?', (limit,))
        results = []
        for row in cursor.fetchall():
            results.append({
                'id': row['id'],
                'timestamp': row['timestamp'],
                'features': json.loads(row['features']),
                'is_anomaly': bool(row['is_anomaly']),
                'confidence': row['confidence'],
                'anomaly_score': row['anomaly_score'],
                'attack_type': row['attack_type'],
                'drift_detected': bool(row['drift_detected']),
                'drift_score': row['drift_score'],
                'llm_explanation': row['llm_explanation']
            })
        conn.close()
        return results
    except Exception as e:
        logger.error(f" : {e}")
        raise HTTPException(status_code=500, detail=f": {str(e)}")
@router.get("/api/system/status")
async def get_system_status():
    try:
        conn = sqlite3.connect('detection_results.db')
        cursor = conn.cursor()
        cursor.execute('SELECT COUNT(*) FROM detection_results')
        total_detections = cursor.fetchone()[0]
        cursor.execute('SELECT COUNT(*) FROM detection_results WHERE is_anomaly = 1')
        anomaly_count = cursor.fetchone()[0]
        cursor.execute('SELECT COUNT(*) FROM detection_results WHERE drift_detected = 1')
        drift_count = cursor.fetchone()[0]
        cursor.execute('SELECT attack_type, COUNT(*) FROM detection_results GROUP BY attack_type')
        attack_types = {row[0]: row[1] for row in cursor.fetchall()}
        conn.close()
        return {
            'total_detections': total_detections,
            'anomaly_count': anomaly_count,
            'drift_count': drift_count,
            'attack_types': attack_types,
            'monitoring_active': network_monitor.is_monitoring,
            'model_initialized': model is not None,
            'llm_initialized': llm_interface is not None
        }
    except Exception as e:
        logger.error(f" : {e}")
        raise HTTPException(status_code=500, detail=f": {str(e)}")
@router.post("/api/simulate/devices")
async def simulate_devices(count: int = 3):
    global detection_history, registered_agents, agent_data_cache
    try:
        import random
        device_names = [f'Server-{chr(65 + i)}' for i in range(count)]
        results = []
        for device_name in device_names:
            if device_name not in registered_agents:
                registered_agents[device_name] = {
                    'status': 'online',
                    'last_seen': datetime.now().isoformat(),
                    'data_count': 0
                }
                agent_data_cache[device_name] = []
            is_anomaly_device = random.random() > 0.7
            if is_anomaly_device:
                features = [
                    random.uniform(0.6, 1.0),
                    random.uniform(0.7, 1.0),
                    random.uniform(0.5, 1.0),
                    random.uniform(0.3, 0.8),
                    random.uniform(0.5, 1.0),
                    random.uniform(0.4, 0.9),
                    random.uniform(0.6, 1.0),
                    random.uniform(0.3, 0.7)
                ]
            else:
                features = [
                    random.uniform(0.05, 0.3),
                    random.uniform(0.05, 0.3),
                    random.uniform(0.05, 0.25),
                    random.uniform(0.05, 0.25),
                    random.uniform(0.05, 0.25),
                    random.uniform(0.05, 0.25),
                    random.uniform(0.3, 0.7),
                    random.uniform(0.4, 0.8)
                ]
            is_anomaly = False
            confidence = 0.5
            attack_type = ''
            if is_anomaly_device:
                is_anomaly = True
                confidence = random.uniform(0.85, 0.99)
                if features[0] > 0.8 or features[1] > 0.8:
                    attack_type = 'DDoS'
                elif features[1] > 0.5 and features[0] < 0.3:
                    attack_type = ''
                elif features[2] > 0.7:
                    attack_type = ''
                else:
                    attack_type = ''
            else:
                confidence = random.uniform(0.8, 0.95)
            result = {
                'agent_name': device_name,
                'timestamp': datetime.now().isoformat(),
                'features': features,
                'is_anomaly': is_anomaly,
                'confidence': confidence,
                'anomaly_score': 1.0 - confidence,
                'attack_type': attack_type,
                'drift_detected': random.random() > 0.8,
                'drift_score': random.uniform(0.0, 0.5),
                'llm_explanation': ''
            }
            detection_history.append(result)
            if len(detection_history) > 500:
                detection_history = detection_history[-500:]
            agent_data_cache[device_name].append(result)
            if len(agent_data_cache[device_name]) > 100:
                agent_data_cache[device_name] = agent_data_cache[device_name][-100:]
            registered_agents[device_name]['last_seen'] = datetime.now().isoformat()
            registered_agents[device_name]['data_count'] += 1
            results.append({'device': device_name, 'is_anomaly': is_anomaly})
        logger.info(f"  {count} ")
        return {'success': True, 'message': f' {count} ', 'devices': results}
    except Exception as e:
        logger.error(f" : {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f": {str(e)}")
@router.get("/api/health")
async def health_check():
    return {
        'status': 'healthy',
        'model_initialized': model is not None,
        'llm_initialized': llm_interface is not None,
        'timestamp': datetime.now().isoformat()
    }
class AgentData(BaseModel):
    agent_name: str = Field(..., description="")
    timestamp: str = Field(..., description="")
    bytes_sent: int = Field(0, description="")
    bytes_recv: int = Field(0, description="")
    packets_sent: int = Field(0, description="")
    packets_recv: int = Field(0, description="")
    active_connections: int = Field(0, description="")
    active_interfaces: int = Field(0, description="")
    features: list = Field([], description="")
@router.get("/api/agents")
async def get_agents():
    try:
        agents = []
        for name, info in registered_agents.items():
            agents.append({
                'name': name,
                'status': info.get('status', 'unknown'),
                'last_seen': info.get('last_seen', ''),
                'data_count': info.get('data_count', 0)
            })
        return {'agents': agents}
    except Exception as e:
        logger.error(f" : {e}")
        raise HTTPException(status_code=500, detail=f": {str(e)}")
@router.get("/api/agents/{agent_name}")
async def get_agent(agent_name: str):
    try:
        if agent_name not in registered_agents:
            raise HTTPException(status_code=404, detail=f" {agent_name} ")
        info = registered_agents[agent_name]
        return {
            'name': agent_name,
            'status': info.get('status', 'unknown'),
            'last_seen': info.get('last_seen', ''),
            'data_count': info.get('data_count', 0)
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f" : {e}")
        raise HTTPException(status_code=500, detail=f": {str(e)}")
@router.post("/api/agent/register")
async def register_agent(agent_name: str):
    try:
        if agent_name in registered_agents:
            return {'success': True, 'message': '', 'registered': False}
        registered_agents[agent_name] = {
            'status': 'online',
            'last_seen': datetime.now().isoformat(),
            'data_count': 0
        }
        agent_data_cache[agent_name] = []
        logger.info(f" : {agent_name}")
        return {'success': True, 'message': '', 'registered': True}
    except Exception as e:
        logger.error(f" : {e}")
        raise HTTPException(status_code=500, detail=f": {str(e)}")
@router.post("/api/agent/data")
async def receive_agent_data(data: AgentData):
    global detection_history
    try:
        agent_name = data.agent_name
        if agent_name not in registered_agents:
            registered_agents[agent_name] = {
                'status': 'online',
                'last_seen': datetime.now().isoformat(),
                'data_count': 0
            }
            agent_data_cache[agent_name] = []
        registered_agents[agent_name]['last_seen'] = datetime.now().isoformat()
        registered_agents[agent_name]['data_count'] += 1
        features = data.features if data.features else []
        if features and len(features) >= 8:
            detection_request = DetectionRequest(features=features, use_llm=False)
            detection_response = await detect_anomaly(detection_request)
            result = {
                'agent_name': agent_name,
                'timestamp': data.timestamp,
                'network_data': {
                    'bytes_sent': data.bytes_sent,
                    'bytes_recv': data.bytes_recv,
                    'packets_sent': data.packets_sent,
                    'packets_recv': data.packets_recv,
                    'active_connections': data.active_connections,
                    'active_interfaces': data.active_interfaces
                },
                'detection_result': {
                    'is_anomaly': detection_response.is_anomaly,
                    'confidence': detection_response.confidence,
                    'attack_type': detection_response.attack_type,
                    'anomaly_score': detection_response.anomaly_score,
                    'drift_detected': detection_response.drift_detected,
                    'drift_score': detection_response.drift_score,
                    'llm_explanation': detection_response.llm_explanation,
                    'features': detection_response.features
                }
            }
            detection_history.append(result)
            if len(detection_history) > 500:
                detection_history = detection_history[-500:]
            agent_data_cache[agent_name].append(result)
            if len(agent_data_cache[agent_name]) > 100:
                agent_data_cache[agent_name] = agent_data_cache[agent_name][-100:]
        return {'success': True}
    except Exception as e:
        logger.error(f" : {e}")
        raise HTTPException(status_code=500, detail=f": {str(e)}")
@router.get("/api/agent/{agent_name}/history")
async def get_agent_history(agent_name: str, limit: int = 20):
    try:
        if agent_name not in agent_data_cache:
            raise HTTPException(status_code=404, detail=f" {agent_name} ")
        data = agent_data_cache[agent_name][-limit:]
        return {'agent_name': agent_name, 'data': data}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f" : {e}")
        raise HTTPException(status_code=500, detail=f": {str(e)}")
@router.delete("/api/agent/{agent_name}")
async def unregister_agent(agent_name: str):
    try:
        if agent_name not in registered_agents:
            raise HTTPException(status_code=404, detail=f" {agent_name} ")
        del registered_agents[agent_name]
        if agent_name in agent_data_cache:
            del agent_data_cache[agent_name]
        logger.info(f" : {agent_name}")
        return {'success': True, 'message': ''}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f" : {e}")
        raise HTTPException(status_code=500, detail=f": {str(e)}")
@router.post("/api/monitor/start")
async def start_monitor():
    try:
        if not network_monitor.is_monitoring:
            network_monitor.start_monitoring()
            return {
                'success': True,
                'message': '',
                'is_monitoring': network_monitor.is_monitoring
            }
        else:
            return {
                'success': True,
                'message': '',
                'is_monitoring': network_monitor.is_monitoring
            }
    except Exception as e:
        logger.error(f" : {e}")
        raise HTTPException(status_code=500, detail=f": {str(e)}")
@router.post("/api/monitor/stop")
async def stop_monitor():
    try:
        if network_monitor.is_monitoring:
            network_monitor.stop_monitoring()
            return {
                'success': True,
                'message': '',
                'is_monitoring': network_monitor.is_monitoring
            }
        else:
            return {
                'success': True,
                'message': '',
                'is_monitoring': network_monitor.is_monitoring
            }
    except Exception as e:
        logger.error(f" : {e}")
        raise HTTPException(status_code=500, detail=f": {str(e)}")
@router.get("/api/export/csv")
async def export_data_csv(start_time: str = None, end_time: str = None):
    try:
        conn = sqlite3.connect('detection_results.db')
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        query = 'SELECT * FROM detection_results'
        params = []
        if start_time:
            query += ' WHERE timestamp >= ?'
            params.append(start_time)
            if end_time:
                query += ' AND timestamp <= ?'
                params.append(end_time)
        elif end_time:
            query += ' WHERE timestamp <= ?'
            params.append(end_time)
        query += ' ORDER BY timestamp DESC'
        cursor.execute(query, params)
        import io
        output = io.StringIO()
        output.write('id,timestamp,is_anomaly,confidence,anomaly_score,attack_type,drift_detected,drift_score\n')
        for row in cursor.fetchall():
            output.write(f"{row['id']},{row['timestamp']},{row['is_anomaly']},{row['confidence']},{row['anomaly_score']},{row['attack_type']},{row['drift_detected']},{row['drift_score']}\n")
        conn.close()
        from fastapi.responses import StreamingResponse
        return StreamingResponse(
            iter([output.getvalue()]),
            media_type="text/csv",
            headers={"Content-Disposition": f"attachment; filename=detection_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"}
        )
    except Exception as e:
        logger.error(f" CSV: {e}")
        raise HTTPException(status_code=500, detail=f"CSV: {str(e)}")
@router.get("/api/export/json")
async def export_data_json(start_time: str = None, end_time: str = None, limit: int = 1000):
    try:
        conn = sqlite3.connect('detection_results.db')
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        query = 'SELECT * FROM detection_results'
        params = []
        if start_time:
            query += ' WHERE timestamp >= ?'
            params.append(start_time)
            if end_time:
                query += ' AND timestamp <= ?'
                params.append(end_time)
        elif end_time:
            query += ' WHERE timestamp <= ?'
            params.append(end_time)
        query += ' ORDER BY timestamp DESC LIMIT ?'
        params.append(limit)
        cursor.execute(query, params)
        results = []
        for row in cursor.fetchall():
            results.append({
                'id': row['id'],
                'timestamp': row['timestamp'],
                'features': json.loads(row['features']),
                'is_anomaly': bool(row['is_anomaly']),
                'confidence': row['confidence'],
                'anomaly_score': row['anomaly_score'],
                'attack_type': row['attack_type'],
                'drift_detected': bool(row['drift_detected']),
                'drift_score': row['drift_score'],
                'llm_explanation': row['llm_explanation']
            })
        conn.close()
        return results
    except Exception as e:
        logger.error(f" JSON: {e}")
        raise HTTPException(status_code=500, detail=f"JSON: {str(e)}")
class AnalysisRequest(BaseModel):
    features: list = Field(None, description="")
    use_llm: bool = Field(True, description="")
    include_recommendations: bool = Field(True, description="")
class ChatRequest(BaseModel):
    message: str = Field(..., description="")
@router.post("/api/chat")
async def chat(request: ChatRequest):
    try:
        if llm_interface and hasattr(llm_interface, 'chat_with_deepseek'):
            deepseek_response = llm_interface.chat_with_deepseek(request.message)
            if deepseek_response:
                logger.info(f"使用 DeepSeek API 响应")
                return {"response": deepseek_response}
        message = request.message.lower()
        if '帮助' in message or '你好' in message or 'hi' in message or 'hello' in message:
            response = "您好！我是网络安全AI助手。我可以帮助您：\n1. 分析网络流量数据\n2. 检测异常行为\n3. 解释安全告警\n4. 提供安全建议\n请问有什么可以帮您的？"
        elif '状态' in message or '网络' in message:
            conn = sqlite3.connect('detection_results.db')
            cursor = conn.cursor()
            cursor.execute('SELECT COUNT(*) FROM detection_results')
            total = cursor.fetchone()[0]
            cursor.execute('SELECT COUNT(*) FROM detection_results WHERE is_anomaly = 1')
            anomaly_count = cursor.fetchone()[0]
            conn.close()
            response = f"当前系统状态：共检测 {total} 次，发现 {anomaly_count} 个异常。系统运行正常，所有检测模块工作正常。"
        elif '异常' in message or '攻击' in message:
            response = "系统使用META-DHPEN元学习模型进行异常检测，可识别DDoS攻击、端口扫描、SQL注入、Web攻击等多种攻击类型。如需查看详情，请查看告警列表或点击AI分析按钮。"
        elif 'ddos' in message:
            response = "DDoS（分布式拒绝服务）攻击利用大量僵尸网络向目标发送海量请求，导致服务不可用。防护措施：使用CDN防护、配置防火墙规则、限制单IP请求频率、启用流量清洗服务。"
        elif '安全' in message or '防护' in message:
            response = "网络安全建议：\n1. 定期更新系统和软件补丁\n2. 配置严格的防火墙规则\n3. 实时监控网络流量异常\n4. 使用HTTPS加密传输\n5. 定期备份重要数据\n6. 加强员工安全意识培训"
        elif '模型' in message or 'AI' in message:
            backend_info = "DeepSeek API" if (llm_interface and getattr(llm_interface, 'backend', '') == 'deepseek') else ""
            response = f"本系统采用META-DHPEN元学习模型进行异常检测，支持5-way 5-shot小样本学习。{backend_info}大模型辅助分析已就绪。"
        elif '历史' in message or '记录' in message:
            conn = sqlite3.connect('detection_results.db')
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute('SELECT * FROM detection_results ORDER BY timestamp DESC LIMIT 5')
            rows = cursor.fetchall()
            conn.close()
            response = "最近5条检测记录：\n\n"
            for row in rows:
                status = "⚠️异常" if row['is_anomaly'] else "✅正常"
                response += f"- {row['timestamp']}: {status} - {row['attack_type']} (置信度: {row['confidence']:.2%})\n"
            if not rows:
                response = "暂无检测记录，请先开始监控。"
        else:
            response = "抱歉，我不太理解您的问题。您可以说：\n- 当前网络状态如何？\n- 什么是DDoS攻击？\n- 如何保护网络安全？\n- 查看历史记录"
        return {"response": response}
    except Exception as e:
        logger.error(f"聊天处理出错: {e}")
        return {"response": f"处理请求时出错: {str(e)}"}
@router.post("/api/analysis/deep")
async def deep_analysis(request: AnalysisRequest = None):
    try:
        if request is None or request.features is None:
            conn = sqlite3.connect('detection_results.db')
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute('SELECT * FROM detection_results ORDER BY timestamp DESC LIMIT 1')
            row = cursor.fetchone()
            if row:
                features = json.loads(row['features'])
                is_anomaly = bool(row['is_anomaly'])
                attack_type = row['attack_type']
                confidence = row['confidence']
            else:
                features = [0.1, 0.2, 0.0, 0.0, 0.5, 0.3, 0.1, 0.2, 5, 2]
                is_anomaly = False
                attack_type = ""
                confidence = 0.95
            conn.close()
        else:
            features = request.features
            features = [0.0 if x is None else x for x in features]
            features = [0.0 if isinstance(x, float) and (np.isnan(x) or np.isinf(x)) else x for x in features]
            features = [float(x) for x in features]
            is_anomaly = False
            attack_type = ""
            confidence = 0.95
            if len(features) >= 2:
                if features[0] > 0.5 or features[1] > 0.5:
                    is_anomaly = True
                    confidence = min(0.95, features[0] + features[1])
                    attack_type = classify_attack_type(features, is_anomaly, confidence)
        analysis_result = {
            'timestamp': datetime.now().isoformat(),
            'features': features,
            'is_anomaly': is_anomaly,
            'attack_type': attack_type,
            'confidence': confidence,
            'analysis': {
                'feature_analysis': {
                    'total_bytes': f": {'' if features[0] > 0.7 else '' if features[0] > 0.4 else ''} ({features[0]:.2f})",
                    'total_packets': f": {'' if features[1] > 0.7 else '' if features[1] > 0.4 else ''} ({features[1]:.2f})",
                    'bytes_sent': f": {'' if features[2] > 0.7 else '' if features[2] > 0.4 else ''} ({features[2]:.2f})",
                    'bytes_recv': f": {'' if features[3] > 0.7 else '' if features[3] > 0.4 else ''} ({features[3]:.2f})"
                },
                'risk_level': '' if is_anomaly and confidence > 0.8 else '' if is_anomaly else '',
                'recommendations': [
                    '',
                    '',
                    '',
                    ''
                ] if is_anomaly else [
                    '',
                    '',
                    ''
                ],
                'confidence_analysis': {
                    'score': confidence,
                    'interpretation': '' if confidence > 0.9 else '' if confidence > 0.7 else ''
                }
            }
        }
        return analysis_result
    except Exception as e:
        logger.error(f" : {e}")
        raise HTTPException(status_code=500, detail=f": {str(e)}")
