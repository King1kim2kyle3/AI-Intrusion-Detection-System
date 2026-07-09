#!/usr/bin/env python3
"""
Data Loading and Preprocessing Module
Handles dataset loading, cleaning, and feature engineering
"""

import logging
from typing import Tuple, Dict
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)


class DataProcessor:
    """Processes and prepares data for model training."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.data = None
        self.scaler = None
        logger.info("DataProcessor initialized")
    
    def load_dataset(self, dataset_type: str) -> pd.DataFrame:
        """Load dataset based on type."""
        logger.info(f"Loading dataset: {dataset_type}")
        
        if dataset_type == 'unsw-nb15':
            return self._load_unsw_nb15()
        elif dataset_type == 'kdd99':
            return self._load_kdd99()
        elif dataset_type == 'cicids2017':
            return self._load_cicids2017()
        else:
            raise ValueError(f"Unknown dataset type: {dataset_type}")
    
    def _load_unsw_nb15(self) -> pd.DataFrame:
        """Load UNSW-NB15 dataset."""
        try:
            df = pd.read_csv('data/raw/UNSW_NB15.csv')
            logger.info(f"Loaded UNSW-NB15: {df.shape[0]} rows, {df.shape[1]} columns")
            return df
        except FileNotFoundError:
            logger.error("UNSW-NB15 dataset not found")
            raise
    
    def _load_kdd99(self) -> pd.DataFrame:
        """Load KDD99 dataset."""
        try:
            df = pd.read_csv('data/raw/kdd99.csv')
            logger.info(f"Loaded KDD99: {df.shape[0]} rows, {df.shape[1]} columns")
            return df
        except FileNotFoundError:
            logger.error("KDD99 dataset not found")
            raise
    
    def _load_cicids2017(self) -> pd.DataFrame:
        """Load CIC-IDS2017 dataset."""
        try:
            df = pd.read_csv('data/raw/cicids2017.csv')
            logger.info(f"Loaded CIC-IDS2017: {df.shape[0]} rows, {df.shape[1]} columns")
            return df
        except FileNotFoundError:
            logger.error("CIC-IDS2017 dataset not found")
            raise
    
    def preprocess(self) -> Tuple[np.ndarray, np.ndarray]:
        """Preprocess data: cleaning, normalization, encoding."""
        logger.info("Starting data preprocessing...")
        
        # Handle missing values
        self.data = self._handle_missing_values(self.data)
        
        # Remove duplicates
        if self.config.get('data', {}).get('remove_duplicates', True):
            initial_shape = self.data.shape[0]
            self.data = self.data.drop_duplicates()
            logger.info(f"Removed {initial_shape - self.data.shape[0]} duplicate rows")
        
        # Separate features and target
        X = self.data.drop('attack_label', axis=1, errors='ignore')
        y = self.data.get('attack_label', None)
        
        # Encode categorical features
        X = self._encode_categorical(X)
        
        # Normalize features
        if self.config.get('data', {}).get('normalize', True):
            X = self._normalize_features(X)
        
        logger.info("Data preprocessing completed")
        return X, y
    
    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in dataset."""
        strategy = self.config.get('data', {}).get('missing_strategy', 'mean')
        
        if strategy == 'drop':
            df = df.dropna()
            logger.info(f"Dropped rows with missing values")
        elif strategy == 'mean':
            for col in df.select_dtypes(include=[np.number]).columns:
                if df[col].isnull().sum() > 0:
                    df[col].fillna(df[col].mean(), inplace=True)
            logger.info(f"Filled numerical missing values with mean")
        elif strategy == 'median':
            for col in df.select_dtypes(include=[np.number]).columns:
                if df[col].isnull().sum() > 0:
                    df[col].fillna(df[col].median(), inplace=True)
            logger.info(f"Filled numerical missing values with median")
        
        return df
    
    def _encode_categorical(self, X: pd.DataFrame) -> pd.DataFrame:
        """Encode categorical features."""
        encoding_type = self.config.get('features', {}).get('encoding', 'labelencoding')
        
        categorical_cols = X.select_dtypes(include=['object']).columns
        
        if encoding_type == 'labelencoding':
            from sklearn.preprocessing import LabelEncoder
            le = LabelEncoder()
            for col in categorical_cols:
                X[col] = le.fit_transform(X[col].astype(str))
            logger.info(f"Applied label encoding to {len(categorical_cols)} categorical features")
        
        elif encoding_type == 'onehotencoding':
            X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)
            logger.info(f"Applied one-hot encoding to {len(categorical_cols)} categorical features")
        
        return X
    
    def _normalize_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Normalize/Scale features."""
        scaler_type = self.config.get('features', {}).get('scaler_type', 'standard')
        
        if scaler_type == 'standard':
            self.scaler = StandardScaler()
        elif scaler_type == 'minmax':
            self.scaler = MinMaxScaler()
        else:
            from sklearn.preprocessing import RobustScaler
            self.scaler = RobustScaler()
        
        X_scaled = self.scaler.fit_transform(X)
        X = pd.DataFrame(X_scaled, columns=X.columns)
        
        logger.info(f"Applied {scaler_type} scaling to features")
        return X
    
    def split_data(self, X, y) -> Tuple:
        """Split data into train/validation/test sets."""
        train_split = self.config.get('data', {}).get('train_split', 0.8)
        val_split = self.config.get('data', {}).get('validation_split', 0.1)
        test_split = self.config.get('data', {}).get('test_split', 0.1)
        
        # First split: train vs temp (val + test)
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=(1 - train_split), random_state=42, stratify=y
        )
        
        # Second split: val vs test
        val_ratio = val_split / (val_split + test_split)
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=(1 - val_ratio), random_state=42, stratify=y_temp
        )
        
        logger.info(f"Data split - Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def save_processed_data(self, output_path: str):
        """Save preprocessed data to disk."""
        import os
        os.makedirs(output_path, exist_ok=True)
        
        # Save data
        if self.data is not None:
            self.data.to_csv(f"{output_path}/processed_data.csv", index=False)
            logger.info(f"Saved processed data to {output_path}/processed_data.csv")
        
        # Save scaler
        if self.scaler is not None:
            import joblib
            joblib.dump(self.scaler, f"{output_path}/scaler.pkl")
            logger.info(f"Saved scaler to {output_path}/scaler.pkl")
