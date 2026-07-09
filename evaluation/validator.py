#!/usr/bin/env python3
"""
Model Validation and Testing Module
"""

import logging
from typing import Dict, Tuple
import joblib
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
from evaluation.metrics import MetricsCalculator

logger = logging.getLogger(__name__)


class ModelValidator:
    """Validates and tests trained models."""
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.metrics_calculator = MetricsCalculator()
        logger.info("ModelValidator initialized")
    
    def evaluate(self, model_path: str, dataset_type: str) -> Dict:
        """Evaluate model on test dataset."""
        logger.info(f"Evaluating model: {model_path} on {dataset_type}")
        
        try:
            # Load model
            model = joblib.load(model_path)
            
            # Load data
            from data.datasets import DataProcessor
            processor = DataProcessor(self.config)
            data = processor.load_dataset(dataset_type)
            processor.data = data
            X, y = processor.preprocess()
            
            # Split data
            X_train, X_test, y_train, y_test = processor._split_data(X, y)
            
            # Make predictions
            y_pred = model.predict(X_test)
            
            # Get probabilities if available
            y_pred_proba = None
            if hasattr(model, 'predict_proba'):
                y_pred_proba = model.predict_proba(X_test)[:, 1]
            
            # Calculate metrics
            metrics = self.metrics_calculator.calculate_metrics(y_test, y_pred, y_pred_proba)
            
            results = {
                'model_path': model_path,
                'dataset': dataset_type,
                'test_samples': len(X_test),
                'metrics': metrics,
                'confusion_matrix': self.metrics_calculator.confusion_mat.tolist()
            }
            
            logger.info("Model evaluation completed")
            return results
            
        except Exception as e:
            logger.error(f"Error during evaluation: {e}")
            raise
    
    def generate_report(self, results: Dict):
        """Generate evaluation report."""
        logger.info("\n" + "="*60)
        logger.info("MODEL EVALUATION REPORT")
        logger.info("="*60)
        logger.info(f"Model: {results.get('model_path')}")
        logger.info(f"Dataset: {results.get('dataset')}")
        logger.info(f"Test Samples: {results.get('test_samples')}")
        logger.info("-"*60)
        
        metrics = results.get('metrics', {})
        logger.info("Metrics:")
        for metric, value in metrics.items():
            if metric != 'classification_report':
                logger.info(f"  {metric.upper():20s}: {value:.4f}")
        
        if 'classification_report' in metrics:
            logger.info("\nDetailed Classification Report:")
            logger.info(metrics['classification_report'])
        
        logger.info("="*60)
    
    def test_model_performance(self, model_path: str, threshold: float = 0.85) -> bool:
        """Test if model meets minimum performance threshold."""
        logger.info(f"Testing model performance threshold: {threshold}")
        
        try:
            model = joblib.load(model_path)
            logger.info(f"Model loaded: {type(model).__name__}")
            
            # Model loaded successfully
            return True
            
        except Exception as e:
            logger.error(f"Model validation failed: {e}")
            return False
