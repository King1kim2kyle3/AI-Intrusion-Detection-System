#!/usr/bin/env python3
"""
Model Evaluation and Metrics Module
Compute and track model performance metrics
"""

import logging
import numpy as np
from typing import Dict, Tuple
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    precision_recall_curve, roc_curve, auc
)
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)


class MetricsCalculator:
    """Calculate and visualize model performance metrics."""
    
    def __init__(self):
        self.metrics = {}
        self.confusion_mat = None
        logger.info("MetricsCalculator initialized")
    
    def calculate_metrics(self, y_true, y_pred, y_pred_proba=None) -> Dict:
        """Calculate comprehensive performance metrics."""
        logger.info("Calculating metrics...")
        
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
            'recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
            'f1_score': f1_score(y_true, y_pred, average='weighted', zero_division=0),
        }
        
        # ROC-AUC if probabilities available
        if y_pred_proba is not None:
            try:
                metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba)
            except Exception as e:
                logger.warning(f"Could not calculate ROC-AUC: {e}")
        
        # Confusion matrix
        self.confusion_mat = confusion_matrix(y_true, y_pred)
        
        # Classification report
        metrics['classification_report'] = classification_report(y_true, y_pred)
        
        self.metrics = metrics
        logger.info(f"Metrics calculated: {metrics}")
        return metrics
    
    def print_metrics(self):
        """Print metrics in readable format."""
        logger.info("=" * 50)
        logger.info("MODEL PERFORMANCE METRICS")
        logger.info("=" * 50)
        
        for metric, value in self.metrics.items():
            if metric != 'classification_report':
                logger.info(f"{metric.upper():20s}: {value:.4f}")
        
        logger.info("\n" + self.metrics.get('classification_report', ''))
        logger.info("=" * 50)
    
    def plot_confusion_matrix(self, output_path: str = None):
        """Plot confusion matrix."""
        if self.confusion_mat is None:
            logger.warning("Confusion matrix not calculated")
            return
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(self.confusion_mat, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        
        if output_path:
            plt.savefig(output_path)
            logger.info(f"Confusion matrix saved to {output_path}")
        else:
            plt.show()
    
    def plot_roc_curve(self, y_true, y_pred_proba, output_path: str = None):
        """Plot ROC curve."""
        try:
            fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
            roc_auc = auc(fpr, tpr)
            
            plt.figure(figsize=(8, 6))
            plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('ROC Curve')
            plt.legend(loc="lower right")
            
            if output_path:
                plt.savefig(output_path)
                logger.info(f"ROC curve saved to {output_path}")
            else:
                plt.show()
                
        except Exception as e:
            logger.error(f"Error plotting ROC curve: {e}")
    
    def plot_precision_recall_curve(self, y_true, y_pred_proba, output_path: str = None):
        """Plot Precision-Recall curve."""
        try:
            precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
            pr_auc = auc(recall, precision)
            
            plt.figure(figsize=(8, 6))
            plt.plot(recall, precision, color='blue', lw=2, label=f'PR curve (AUC = {pr_auc:.2f})')
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.title('Precision-Recall Curve')
            plt.legend(loc="lower left")
            plt.grid(True)
            
            if output_path:
                plt.savefig(output_path)
                logger.info(f"PR curve saved to {output_path}")
            else:
                plt.show()
                
        except Exception as e:
            logger.error(f"Error plotting PR curve: {e}")
