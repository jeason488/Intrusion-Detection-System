import torch
import torch.nn as nn
import logging
import os
import requests
import json
logging.basicConfig(level=logging.INFO, format='%(levelname)s\t%(message)s')
logger = logging.getLogger(__name__)
class LLMAuxiliary:
    def __init__(self, config: dict):
        self.config = config
        self.model_name = config.get('model_name', 'bert-base-uncased')
        self.max_seq_length = config.get('max_seq_length', 128)
        self.device = config.get('device', 'cpu')
        self.use_deepseek = config.get('use_deepseek', False)
        self.deepseek_api_key = config.get('deepseek_api_key', '')
        self.deepseek_api_url = config.get('deepseek_api_url', 'https://api.deepseek.com/v1/chat/completions')
        self.deepseek_model = config.get('deepseek_model', 'deepseek-chat')
        os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
        logger.info(f" HF_ENDPOINT: {os.environ['HF_ENDPOINT']}")
        self.use_real_llm = False
        if self.use_deepseek and self.deepseek_api_key:
            logger.info("  DeepSeek API")
            if self._test_deepseek_connection():
                logger.info("  DeepSeek API ")
                self.use_real_llm = True
                self.backend = 'deepseek'
            else:
                logger.warning("  DeepSeek API ")
        if not self.use_real_llm:
            try:
                from transformers import AutoModel, AutoTokenizer
                logger.info(f" : {self.model_name}")
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                self.model = AutoModel.from_pretrained(self.model_name)
                self.model.to(self.device)
                self.model.eval()
                logger.info(" ")
                self.use_real_llm = True
                self.backend = 'transformers'
            except Exception as e:
                logger.warning(f" : {e}")
                logger.info(" ")
                self.backend = 'rule_based'
                self._init_rule_engine()
    def _test_deepseek_connection(self):
        try:
            headers = {
                'Authorization': f'Bearer {self.deepseek_api_key}',
                'Content-Type': 'application/json'
            }
            data = {
                'model': self.deepseek_model,
                'messages': [{'role': 'user', 'content': 'hello'}],
                'max_tokens': 10
            }
            response = requests.post(self.deepseek_api_url, headers=headers, json=data, timeout=10)
            return response.status_code == 200
        except Exception as e:
            logger.error(f" DeepSeek: {e}")
            return False
    def _call_deepseek_api(self, messages, max_tokens=512, temperature=0.7):
        try:
            headers = {
                'Authorization': f'Bearer {self.deepseek_api_key}',
                'Content-Type': 'application/json'
            }
            data = {
                'model': self.deepseek_model,
                'messages': messages,
                'max_tokens': max_tokens,
                'temperature': temperature
            }
            response = requests.post(self.deepseek_api_url, headers=headers, json=data, timeout=30)
            if response.status_code == 200:
                result = response.json()
                return result['choices'][0]['message']['content']
            else:
                logger.error(f" DeepSeek API: {response.status_code} - {response.text}")
                return None
        except Exception as e:
            logger.error(f" DeepSeek API: {e}")
            return None
    def _init_rule_engine(self):
        self.attack_type_mapping = {
            0: {"name": "", "severity": "", "description": ""},
            1: {"name": "", "severity": "", "description": ""},
            2: {"name": "", "severity": "", "description": ""}
        }
    def generate_detection_explanation(self, features, prediction, confidence, model_info=None):
        if self.use_real_llm:
            return self._generate_llm_explanation(features, prediction, confidence)
        else:
            return self._generate_rule_based_explanation(features, prediction, confidence)
    def _generate_llm_explanation(self, features, prediction, confidence):
        if self.backend == 'deepseek':
            return self._generate_deepseek_explanation(features, prediction, confidence)
        else:
            prompt = f":\n"
            prompt += f": {features[:5]}...\n"
            prompt += f": {'' if prediction == 1 else ''}\n"
            prompt += f": {confidence:.2f}\n"
            prompt += ""
            inputs = self.tokenizer(prompt, return_tensors='pt', max_length=512, truncation=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = self.model(**inputs)
                return self._generate_rule_based_explanation(features, prediction, confidence)
    def _generate_deepseek_explanation(self, features, prediction, confidence):
        system_prompt = "你是一个网络安全分析专家。请分析网络流量数据并提供安全评估。"
        user_prompt = f"特征数据: {features[:5]}, 检测结果: {'异常' if prediction == 1 else '正常'}, 置信度: {confidence:.2f}"
        messages = [
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': user_prompt}
        ]
        response = self._call_deepseek_api(messages, max_tokens=1024)
        if response:
            return {
                "detection_result": "异常" if prediction == 1 else "正常",
                "confidence": f"{confidence:.2%}",
                "analysis": [{"llm_analysis": response}],
                "suggestions": ["AI分析建议：请查看详细报告"],
                "threat_level": "高危" if prediction == 1 else "安全"
            }
        else:
            return self._generate_rule_based_explanation(features, prediction, confidence)
    def chat_with_deepseek(self, message, history=None):
        if self.backend != 'deepseek':
            return None
        system_prompt = "你是一个网络安全助手，请用中文回答用户关于网络安全的问题。"
        messages = [{'role': 'system', 'content': system_prompt}]
        if history:
            for item in history:
                messages.append({'role': 'user', 'content': item['user']})
                messages.append({'role': 'assistant', 'content': item['assistant']})
        messages.append({'role': 'user', 'content': message})
        return self._call_deepseek_api(messages, max_tokens=1024)
    def _generate_rule_based_explanation(self, features, prediction, confidence):
        result = {
            "detection_result": "" if prediction == 0 else "",
            "confidence": f"{confidence:.2%}",
            "analysis": [],
            "suggestions": [],
            "threat_level": "" if prediction == 0 else ""
        }
        feature_analysis = self._analyze_features(features)
        result["analysis"].append(feature_analysis)
        if prediction == 1:
            result["threat_level"] = self._assess_threat_level(features, confidence)
            result["suggestions"] = self._generate_defense_suggestions(features)
            result["attack_type"] = self._infer_attack_type(features)
        else:
            result["suggestions"].append("")
        return result
    def _analyze_features(self, features):
        if not isinstance(features, (list, tuple)):
            features = list(features)
        analysis = {
            "packet_count": int(sum(features[4:6])) if len(features) > 6 else 0,
            "byte_transfer": int(sum(features[6:8])) if len(features) > 8 else 0,
            "connection_duration": float(features[0]) if len(features) > 0 else 0.0
        }
        if analysis["packet_count"] > 1000:
            analysis["note"] = ""
        if analysis["byte_transfer"] > 100000:
            analysis["note"] = ""
        return analysis
    def _assess_threat_level(self, features, confidence):
        if confidence > 0.9:
            return ""
        elif confidence > 0.7:
            return ""
        else:
            return ""
    def _infer_attack_type(self, features):
        if len(features) < 10:
            return ""
        duration = features[0] if len(features) > 0 else 0
        src_bytes = features[6] if len(features) > 6 else 0
        dst_bytes = features[7] if len(features) > 7 else 0
        if src_bytes > 1000000 and duration < 1:
            return "DoS - "
        elif dst_bytes > 1000000 and duration < 1:
            return " - "
        elif duration > 1000:
            return " - "
        else:
            return ""
    def _generate_defense_suggestions(self, features):
        suggestions = []
        if len(features) > 6:
            src_pkts = features[4] if features[4] else 0
            if src_pkts > 500:
                suggestions.append("DDoS")
                suggestions.append("IP")
        if len(features) > 8:
            src_bytes = features[6] if features[6] else 0
            if src_bytes > 500000:
                suggestions.append("")
                suggestions.append("")
        if not suggestions:
            suggestions.append("")
            suggestions.append("")
        return suggestions
    def enrich_detection_result(self, base_result, detection_metadata=None):
        enriched = base_result.copy()
        import datetime
        enriched["timestamp"] = datetime.datetime.now().isoformat()
        enriched["threat_intelligence"] = {
            "category": "",
            "detection_method": " + ",
            "model_version": "1.0.0"
        }
        if detection_metadata:
            enriched["context"] = {
                "drift_detected": detection_metadata.get("is_drift", False),
                "adaptation_performed": detection_metadata.get("adapted", False)
            }
        return enriched
    def summarize_attack_pattern(self, historical_detections):
        if not historical_detections:
            return ""
        attack_count = sum(1 for d in historical_detections if d.get("prediction") == 1)
        total = len(historical_detections)
        summary = f" {total}  {attack_count} "
        summary += f"\n{attack_count/total:.1%}" if total > 0 else ""
        return summary
