#!/usr/bin/env python3
"""
Real-time Intrusion Detection Engine
Provides methods for detecting intrusions in network traffic
"""

import logging
import numpy as np
import pandas as pd
import joblib
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import yaml

logger = logging.getLogger(__name__)


class IntrusionDetector:
    """Real-time intrusion detection engine."""
    
    def __init__(self, model_path: str, config: Dict = None):
        self.model_path = model_path
        self.config = config or {}
        self.model = None
        self.scaler = None
        self.feature_names = None
        self.load_model()
        logger.info(f"IntrusionDetector initialized with model: {model_path}")
    
    def load_model(self):
        """Load trained model from disk."""
        try:
            self.model = joblib.load(self.model_path)
            logger.info(f"Model loaded successfully from {self.model_path}")
        except FileNotFoundError:
            logger.error(f"Model file not found: {self.model_path}")
            raise
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
    def detect(self, features: np.ndarray) -> Dict:
        """Detect intrusion in network packet/flow.
        
        Args:
            features: Array of network features
            
        Returns:
            Dict with prediction, confidence, and attack type
        """
        if self.model is None:
            raise RuntimeError("Model not loaded")
        
        try:
            # Preprocess features
            features = self._preprocess_features(features)
            
            # Make prediction
            if hasattr(self.model, 'predict_proba'):
                prediction = self.model.predict(features)[0]
                confidence = np.max(self.model.predict_proba(features))
            else:
                prediction = self.model.predict(features)[0]
                confidence = 0.5
            
            # Determine if intrusion
            threshold = self.config.get('detection', {}).get('anomaly_threshold', 0.5)
            is_intrusion = prediction > threshold
            
            result = {
                'timestamp': datetime.now().isoformat(),
                'is_intrusion': bool(is_intrusion),
                'prediction': float(prediction),
                'confidence': float(confidence),
                'threat_level': self._classify_threat_level(confidence),
                'model': type(self.model).__name__
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error during detection: {e}")
            return {'error': str(e), 'is_intrusion': False}
    
    def detect_batch(self, features: np.ndarray) -> List[Dict]:
        """Detect intrusions in batch of network packets.
        
        Args:
            features: Array of network features (N x M)
            
        Returns:
            List of detection results
        """
        results = []
        for feature_vector in features:
            result = self.detect(feature_vector)
            results.append(result)
        return results
    
    def _preprocess_features(self, features: np.ndarray) -> np.ndarray:
        """Preprocess features before prediction."""
        # Ensure proper shape
        if len(features.shape) == 1:
            features = features.reshape(1, -1)
        
        # Apply scaling if available
        if self.scaler is not None:
            features = self.scaler.transform(features)
        
        return features
    
    def _classify_threat_level(self, confidence: float) -> str:
        """Classify threat level based on confidence."""
        if confidence >= 0.9:
            return "CRITICAL"
        elif confidence >= 0.7:
            return "HIGH"
        elif confidence >= 0.5:
            return "MEDIUM"
        else:
            return "LOW"
    
    def start_detection(self, mode: str = 'real-time'):
        """Start detection in specified mode."""
        logger.info(f"Starting detection in {mode} mode")
        
        if mode == 'real-time':
            self._real_time_detection()
        elif mode == 'batch':
            self._batch_detection()
        elif mode == 'streaming':
            self._streaming_detection()
        else:
            logger.error(f"Unknown detection mode: {mode}")
    
    def _real_time_detection(self):
        """Real-time packet detection."""
        logger.info("Real-time detection started")
        # Implementation for packet capture and detection
        pass
    
    def _batch_detection(self):
        """Batch detection from files."""
        logger.info("Batch detection started")
        # Implementation for batch processing
        pass
    
    def _streaming_detection(self):
        """Streaming detection from Kafka/RabbitMQ."""
        logger.info("Streaming detection started")
        # Implementation for streaming processing
        pass
