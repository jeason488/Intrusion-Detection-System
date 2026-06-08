import numpy as np
from scipy import stats
import logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s\t%(message)s')
logger = logging.getLogger(__name__)
class DriftDetector:
    def __init__(self, config: dict):
        self.config = config
        self.bandwidth = config.get('bandwidth', 0.1)
        self.threshold = config.get('threshold', 0.5)
        self.window_size = config.get('window_size', 100)
        self.history_data = []
    def detect_drift(self, current_data, reference_data=None):
        if reference_data is None:
            if len(self.history_data) < self.window_size:
                self.history_data.extend(current_data)
                if len(self.history_data) > self.window_size:
                    self.history_data = self.history_data[-self.window_size:]
                return False, 0.0
            else:
                reference_data = self.history_data
        self.history_data.extend(current_data)
        if len(self.history_data) > self.window_size:
            self.history_data = self.history_data[-self.window_size:]
        drift_score = self._calculate_drift_score(current_data, reference_data)
        is_drift = drift_score > self.threshold
        if is_drift:
            logger.info(f" : {drift_score:.4f}")
        return is_drift, drift_score
    def _calculate_drift_score(self, current_data, reference_data):
        current_data = np.array(current_data)
        reference_data = np.array(reference_data)
        if current_data.ndim == 2 and reference_data.ndim == 2:
            scores = []
            for i in range(current_data.shape[1]):
                try:
                    _, p_value = stats.kstest(current_data[:, i], reference_data[:, i])
                    scores.append(1 - p_value)
                except:
                    scores.append(0.0)
            drift_score = np.mean(scores)
        else:
            try:
                _, p_value = stats.kstest(current_data, reference_data)
                drift_score = 1 - p_value
            except:
                drift_score = 0.0
        return drift_score
