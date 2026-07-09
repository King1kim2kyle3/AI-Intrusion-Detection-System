#!/usr/bin/env python3
"""
Unit Tests for IDS System
"""

import pytest
import numpy as np
from detection.detector import IntrusionDetector
from alerts.alert_manager import AlertManager
from alerts.response_handler import ResponseHandler


class TestIntrusionDetector:
    """Test cases for IntrusionDetector."""
    
    @pytest.fixture
    def detector(self):
        """Create detector instance."""
        config = {'detection': {'anomaly_threshold': 0.5}}
        # Mock model for testing
        try:
            return IntrusionDetector('models/saved_models/ids_model.pkl', config)
        except:
            pytest.skip("Model not available")
    
    def test_detection_output_format(self, detector):
        """Test detection output has required fields."""
        features = np.random.rand(1, 50)
        result = detector.detect(features)
        
        assert 'is_intrusion' in result
        assert 'confidence' in result
        assert 'threat_level' in result
        assert 'timestamp' in result


class TestAlertManager:
    """Test cases for AlertManager."""
    
    @pytest.fixture
    def alert_manager(self):
        """Create AlertManager instance."""
        config = {'alerts': {'enabled': True, 'channels': {}}}
        return AlertManager(config)
    
    def test_alert_creation(self, alert_manager):
        """Test alert creation from detection result."""
        detection_result = {
            'is_intrusion': True,
            'confidence': 0.95,
            'threat_level': 'HIGH'
        }
        
        alert = alert_manager.create_alert(detection_result)
        
        assert alert['id'] is not None
        assert alert['severity'] == 'HIGH'
        assert alert['timestamp'] is not None


class TestResponseHandler:
    """Test cases for ResponseHandler."""
    
    @pytest.fixture
    def response_handler(self):
        """Create ResponseHandler instance."""
        config = {
            'response': {
                'enabled': True,
                'response_actions': []
            }
        }
        return ResponseHandler(config)
    
    def test_response_handler_initialization(self, response_handler):
        """Test ResponseHandler initialization."""
        assert response_handler is not None
        assert isinstance(response_handler.blocked_ips, set)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
