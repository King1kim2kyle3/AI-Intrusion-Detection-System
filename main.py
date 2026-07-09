#!/usr/bin/env python3
"""
AI-Powered Intrusion Detection System
Main Entry Point

This module serves as the entry point for the IDS system and provides
CLI commands for training, detection, evaluation, and more.
"""

import sys
import argparse
import logging
from pathlib import Path
from typing import Optional
import yaml

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ConfigManager:
    """Manages configuration loading and validation."""
    
    def __init__(self, config_path: str = 'config/config.yaml'):
        self.config_path = Path(config_path)
        self.config = self._load_config()
    
    def _load_config(self) -> dict:
        """Load configuration from YAML file."""
        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
            logger.info(f"Configuration loaded from {self.config_path}")
            return config
        except FileNotFoundError:
            logger.error(f"Configuration file not found: {self.config_path}")
            sys.exit(1)
        except yaml.YAMLError as e:
            logger.error(f"Error parsing configuration: {e}")
            sys.exit(1)
    
    def get(self, key: str, default=None):
        """Get configuration value."""
        keys = key.split('.')
        value = self.config
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
            else:
                return default
        return value if value is not None else default


class IDSSystem:
    """Main IDS System class."""
    
    def __init__(self, config_path: str = 'config/config.yaml'):
        self.config_manager = ConfigManager(config_path)
        self.config = self.config_manager.config
        logger.info(f"IDS System initialized: {self.config.get('system', {}).get('name')}")
    
    def train_models(self, args):
        """Train ML models on intrusion detection dataset."""
        logger.info("Starting model training...")
        
        dataset = args.dataset or self.config_manager.get('data.dataset_type', 'unsw-nb15')
        model_type = args.model or 'ensemble'
        epochs = args.epochs or self.config_manager.get('training.epochs', 50)
        
        logger.info(f"Training configuration:")
        logger.info(f"  Dataset: {dataset}")
        logger.info(f"  Model Type: {model_type}")
        logger.info(f"  Epochs: {epochs}")
        
        try:
            # Import training module dynamically
            from models.training import ModelTrainer
            
            trainer = ModelTrainer(self.config)
            trainer.load_data(dataset)
            trainer.train(model_type=model_type, epochs=epochs)
            trainer.evaluate()
            trainer.save_model()
            
            logger.info("Model training completed successfully!")
        except ImportError as e:
            logger.error(f"Failed to import training module: {e}")
            logger.info("Please ensure all dependencies are installed: pip install -r requirements.txt")
            sys.exit(1)
        except Exception as e:
            logger.error(f"Error during model training: {e}")
            sys.exit(1)
    
    def run_detection(self, args):
        """Run real-time intrusion detection."""
        logger.info("Starting real-time intrusion detection...")
        
        model_path = args.model or 'models/saved_models/ids_model.pkl'
        mode = args.mode or self.config_manager.get('detection.mode', 'real-time')
        
        logger.info(f"Detection configuration:")
        logger.info(f"  Model: {model_path}")
        logger.info(f"  Mode: {mode}")
        
        try:
            from detection.detector import IntrusionDetector
            
            detector = IntrusionDetector(model_path, self.config)
            detector.start_detection(mode=mode)
            
        except ImportError as e:
            logger.error(f"Failed to import detection module: {e}")
            sys.exit(1)
        except Exception as e:
            logger.error(f"Error during detection: {e}")
            sys.exit(1)
    
    def evaluate_models(self, args):
        """Evaluate model performance."""
        logger.info("Starting model evaluation...")
        
        model_path = args.model or 'models/saved_models/ids_model.pkl'
        dataset = args.dataset or self.config_manager.get('data.dataset_type', 'unsw-nb15')
        
        logger.info(f"Evaluation configuration:")
        logger.info(f"  Model: {model_path}")
        logger.info(f"  Dataset: {dataset}")
        
        try:
            from evaluation.validator import ModelValidator
            
            validator = ModelValidator(self.config)
            results = validator.evaluate(model_path, dataset)
            validator.generate_report(results)
            
            logger.info("Model evaluation completed successfully!")
        except ImportError as e:
            logger.error(f"Failed to import evaluation module: {e}")
            sys.exit(1)
        except Exception as e:
            logger.error(f"Error during evaluation: {e}")
            sys.exit(1)
    
    def preprocess_data(self, args):
        """Preprocess and prepare dataset."""
        logger.info("Starting data preprocessing...")
        
        dataset = args.dataset or self.config_manager.get('data.dataset_type', 'unsw-nb15')
        output_path = args.output or self.config_manager.get('data.processed_path', 'data/processed/')
        
        logger.info(f"Preprocessing configuration:")
        logger.info(f"  Dataset: {dataset}")
        logger.info(f"  Output Path: {output_path}")
        
        try:
            from data.datasets import DataProcessor
            
            processor = DataProcessor(self.config)
            processor.load_dataset(dataset)
            processor.preprocess()
            processor.save_processed_data(output_path)
            
            logger.info("Data preprocessing completed successfully!")
        except ImportError as e:
            logger.error(f"Failed to import data module: {e}")
            sys.exit(1)
        except Exception as e:
            logger.error(f"Error during preprocessing: {e}")
            sys.exit(1)
    
    def start_api_server(self, args):
        """Start REST API server."""
        logger.info("Starting API server...")
        
        host = args.host or self.config_manager.get('deployment.host', '0.0.0.0')
        port = args.port or self.config_manager.get('deployment.port', 5000)
        debug = args.debug or self.config_manager.get('system.debug', False)
        
        logger.info(f"API Server configuration:")
        logger.info(f"  Host: {host}")
        logger.info(f"  Port: {port}")
        logger.info(f"  Debug: {debug}")
        
        try:
            from flask import Flask
            from detection.detector import IntrusionDetector
            
            app = Flask(__name__)
            detector = IntrusionDetector(self.config)
            
            @app.route('/api/v1/predict', methods=['POST'])
            def predict():
                """Predict intrusion on network packet."""
                # Implementation will be in api module
                pass
            
            @app.route('/api/v1/status', methods=['GET'])
            def status():
                """Get system status."""
                return {'status': 'running', 'system': 'IDS'}
            
            logger.info(f"API Server running on http://{host}:{port}")
            app.run(host=host, port=port, debug=debug)
            
        except ImportError as e:
            logger.error(f"Failed to import required modules: {e}")
            logger.info("Please ensure Flask is installed: pip install flask")
            sys.exit(1)
        except Exception as e:
            logger.error(f"Error starting API server: {e}")
            sys.exit(1)
    
    def show_status(self, args):
        """Show system status and statistics."""
        logger.info("System Status Report")
        logger.info("-" * 50)
        logger.info(f"System Name: {self.config_manager.get('system.name')}")
        logger.info(f"Version: {self.config_manager.get('system.version')}")
        logger.info(f"Mode: {self.config_manager.get('system.mode')}")
        logger.info(f"Dataset Type: {self.config_manager.get('data.dataset_type')}")
        logger.info(f"Alerts Enabled: {self.config_manager.get('alerts.enabled')}")
        logger.info(f"Auto Response Enabled: {self.config_manager.get('response.auto_response')}")
        logger.info("-" * 50)


def create_parser() -> argparse.ArgumentParser:
    """Create and configure argument parser."""
    parser = argparse.ArgumentParser(
        description='AI-Powered Intrusion Detection System',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train models
  python main.py train --dataset unsw-nb15 --model xgboost --epochs 50
  
  # Run real-time detection
  python main.py detect --mode real-time --model saved_models/ids_model.pkl
  
  # Evaluate model performance
  python main.py evaluate --model saved_models/ids_model.pkl
  
  # Preprocess data
  python main.py preprocess --dataset unsw-nb15 --output data/processed/
  
  # Start API server
  python main.py api --host 0.0.0.0 --port 5000
  
  # Show system status
  python main.py status
        """
    )
    
    parser.add_argument(
        '--config',
        default='config/config.yaml',
        help='Path to configuration file (default: config/config.yaml)'
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train ML models')
    train_parser.add_argument('--dataset', help='Dataset to use (kdd99, unsw-nb15, cicids2017)')
    train_parser.add_argument('--model', help='Model type (random_forest, xgboost, neural_network, ensemble)')
    train_parser.add_argument('--epochs', type=int, help='Number of training epochs')
    train_parser.set_defaults(func='train_models')
    
    # Detection command
    detect_parser = subparsers.add_parser('detect', help='Run real-time detection')
    detect_parser.add_argument('--mode', help='Detection mode (real-time, batch, streaming)')
    detect_parser.add_argument('--model', help='Path to trained model')
    detect_parser.set_defaults(func='run_detection')
    
    # Evaluation command
    eval_parser = subparsers.add_parser('evaluate', help='Evaluate model performance')
    eval_parser.add_argument('--model', help='Path to trained model')
    eval_parser.add_argument('--dataset', help='Dataset to evaluate on')
    eval_parser.set_defaults(func='evaluate_models')
    
    # Preprocessing command
    preprocess_parser = subparsers.add_parser('preprocess', help='Preprocess dataset')
    preprocess_parser.add_argument('--dataset', help='Dataset to preprocess')
    preprocess_parser.add_argument('--output', help='Output path for processed data')
    preprocess_parser.set_defaults(func='preprocess_data')
    
    # API server command
    api_parser = subparsers.add_parser('api', help='Start API server')
    api_parser.add_argument('--host', help='Server host')
    api_parser.add_argument('--port', type=int, help='Server port')
    api_parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    api_parser.set_defaults(func='start_api_server')
    
    # Status command
    status_parser = subparsers.add_parser('status', help='Show system status')
    status_parser.set_defaults(func='show_status')
    
    return parser


def main():
    """Main entry point."""
    parser = create_parser()
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    # Initialize IDS system
    ids_system = IDSSystem(config_path=args.config)
    
    # Execute command
    command_func = getattr(ids_system, args.func, None)
    if command_func:
        command_func(args)
    else:
        logger.error(f"Unknown command: {args.command}")
        sys.exit(1)


if __name__ == '__main__':
    main()
