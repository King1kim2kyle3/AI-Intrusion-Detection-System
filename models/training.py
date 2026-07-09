#!/usr/bin/env python3
"""
Model Training Module
Train and evaluate machine learning models for intrusion detection
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, Tuple
import joblib
import yaml
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)

logger = logging.getLogger(__name__)


class ModelTrainer:
    """Trains and manages ML models for IDS."""
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.model = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.history = {}
        logger.info("ModelTrainer initialized")
    
    def load_data(self, dataset_type: str):
        """Load dataset for training."""
        logger.info(f"Loading dataset: {dataset_type}")
        
        try:
            from data.datasets import DataProcessor
            
            processor = DataProcessor(self.config)
            data = processor.load_dataset(dataset_type)
            processor.data = data
            X, y = processor.preprocess()
            
            self.X_train, self.X_test, self.y_train, self.y_test = self._split_data(X, y)
            
            logger.info(f"Data loaded successfully")
            logger.info(f"Training set size: {len(self.X_train)}")
            logger.info(f"Test set size: {len(self.X_test)}")
            
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise
    
    def _split_data(self, X, y) -> Tuple:
        """Split data into train and test sets."""
        from sklearn.model_selection import train_test_split
        
        test_size = self.config.get('data', {}).get('test_split', 0.2)
        random_state = self.config.get('training', {}).get('random_seed', 42)
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        return X_train, X_test, y_train, y_test
    
    def train(self, model_type: str = 'random_forest', epochs: int = 50):
        """Train ML model."""
        logger.info(f"Training {model_type} model for {epochs} epochs")
        
        if model_type == 'random_forest':
            self._train_random_forest()
        elif model_type == 'xgboost':
            self._train_xgboost()
        elif model_type == 'neural_network':
            self._train_neural_network(epochs)
        elif model_type == 'ensemble':
            self._train_ensemble()
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        logger.info(f"Model training completed")
    
    def _train_random_forest(self):
        """Train Random Forest model."""
        logger.info("Training Random Forest model...")
        
        config = self.config.get('models', {}).get('random_forest', {})
        
        self.model = RandomForestClassifier(
            n_estimators=config.get('n_estimators', 100),
            max_depth=config.get('max_depth', 15),
            min_samples_split=config.get('min_samples_split', 5),
            min_samples_leaf=config.get('min_samples_leaf', 2),
            random_state=config.get('random_state', 42),
            n_jobs=config.get('n_jobs', -1),
            class_weight=config.get('class_weight', 'balanced')
        )
        
        self.model.fit(self.X_train, self.y_train)
        logger.info("Random Forest training completed")
    
    def _train_xgboost(self):
        """Train XGBoost model."""
        logger.info("Training XGBoost model...")
        
        try:
            import xgboost as xgb
            
            config = self.config.get('models', {}).get('xgboost', {})
            
            self.model = xgb.XGBClassifier(
                n_estimators=config.get('n_estimators', 100),
                max_depth=config.get('max_depth', 7),
                learning_rate=config.get('learning_rate', 0.1),
                subsample=config.get('subsample', 0.8),
                colsample_bytree=config.get('colsample_bytree', 0.8),
                random_state=config.get('random_state', 42),
                eval_metric=config.get('eval_metric', 'logloss'),
                tree_method=config.get('tree_method', 'hist')
            )
            
            self.model.fit(self.X_train, self.y_train)
            logger.info("XGBoost training completed")
            
        except ImportError:
            logger.error("XGBoost not installed. Install with: pip install xgboost")
            raise
    
    def _train_neural_network(self, epochs: int):
        """Train Neural Network model."""
        logger.info("Training Neural Network model...")
        
        try:
            from tensorflow import keras
            from tensorflow.keras import layers
            
            config = self.config.get('models', {}).get('neural_network', {})
            
            # Build model
            model = keras.Sequential()
            model.add(layers.Input(shape=(self.X_train.shape[1],)))
            model.add(layers.Dropout(config.get('input_dropout', 0.2)))
            
            for units in config.get('hidden_layers', [128, 64, 32]):
                model.add(layers.Dense(units, activation=config.get('hidden_activation', 'relu')))
                if config.get('batch_normalization', True):
                    model.add(layers.BatchNormalization())
                model.add(layers.Dropout(config.get('hidden_dropout', 0.3)))
            
            model.add(layers.Dense(1, activation=config.get('output_activation', 'sigmoid')))
            
            # Compile model
            model.compile(
                optimizer=keras.optimizers.Adam(learning_rate=config.get('learning_rate', 0.001)),
                loss=config.get('loss_function', 'binary_crossentropy'),
                metrics=['accuracy', keras.metrics.Precision(), keras.metrics.Recall()]
            )
            
            # Train model
            self.model = model
            self.history = model.fit(
                self.X_train, self.y_train,
                batch_size=config.get('batch_size', 32),
                epochs=epochs,
                validation_split=config.get('validation_split', 0.2),
                verbose=1
            )
            
            logger.info("Neural Network training completed")
            
        except ImportError:
            logger.error("TensorFlow not installed. Install with: pip install tensorflow")
            raise
    
    def _train_ensemble(self):
        """Train ensemble of models."""
        logger.info("Training ensemble model...")
        
        try:
            from sklearn.ensemble import VotingClassifier
            
            estimators = []
            
            # Random Forest
            if self.config.get('models', {}).get('random_forest', {}).get('enabled', True):
                rf_config = self.config.get('models', {}).get('random_forest', {})
                rf = RandomForestClassifier(
                    n_estimators=rf_config.get('n_estimators', 100),
                    max_depth=rf_config.get('max_depth', 15),
                    random_state=rf_config.get('random_state', 42)
                )
                estimators.append(('rf', rf))
            
            # XGBoost
            if self.config.get('models', {}).get('xgboost', {}).get('enabled', True):
                try:
                    import xgboost as xgb
                    xgb_config = self.config.get('models', {}).get('xgboost', {})
                    xgb_model = xgb.XGBClassifier(
                        n_estimators=xgb_config.get('n_estimators', 100),
                        max_depth=xgb_config.get('max_depth', 7),
                        random_state=xgb_config.get('random_state', 42)
                    )
                    estimators.append(('xgb', xgb_model))
                except ImportError:
                    logger.warning("XGBoost not available for ensemble")
            
            # Create voting classifier
            ensemble_method = self.config.get('models', {}).get('ensemble_method', 'voting')
            self.model = VotingClassifier(estimators=estimators, voting='soft')
            self.model.fit(self.X_train, self.y_train)
            
            logger.info("Ensemble training completed")
            
        except Exception as e:
            logger.error(f"Error training ensemble: {e}")
            raise
    
    def evaluate(self):
        """Evaluate model performance."""
        logger.info("Evaluating model...")
        
        if self.model is None:
            raise RuntimeError("Model not trained")
        
        try:
            y_pred = self.model.predict(self.X_test)
            
            accuracy = accuracy_score(self.y_test, y_pred)
            precision = precision_score(self.y_test, y_pred, average='weighted')
            recall = recall_score(self.y_test, y_pred, average='weighted')
            f1 = f1_score(self.y_test, y_pred, average='weighted')
            
            logger.info(f"Accuracy: {accuracy:.4f}")
            logger.info(f"Precision: {precision:.4f}")
            logger.info(f"Recall: {recall:.4f}")
            logger.info(f"F1-Score: {f1:.4f}")
            
            if hasattr(self.model, 'predict_proba'):
                y_pred_proba = self.model.predict_proba(self.X_test)[:, 1]
                roc_auc = roc_auc_score(self.y_test, y_pred_proba)
                logger.info(f"ROC-AUC: {roc_auc:.4f}")
            
            logger.info("\nClassification Report:")
            logger.info(classification_report(self.y_test, y_pred))
            
            return {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1
            }
            
        except Exception as e:
            logger.error(f"Error during evaluation: {e}")
            raise
    
    def save_model(self, path: str = None):
        """Save trained model to disk."""
        if self.model is None:
            raise RuntimeError("No model to save")
        
        path = path or self.config.get('training', {}).get('model_save_path', 'models/saved_models/ids_model.pkl')
        
        try:
            joblib.dump(self.model, path)
            logger.info(f"Model saved to {path}")
        except Exception as e:
            logger.error(f"Error saving model: {e}")
            raise
