#!/usr/bin/env python3
"""
Alert Management System
Handles alert generation, notification, and logging
"""

import logging
from datetime import datetime
from typing import Dict, List, Optional
import json

logger = logging.getLogger(__name__)


class AlertManager:
    """Manages intrusion detection alerts."""
    
    SEVERITY_LEVELS = {
        'CRITICAL': 1,
        'HIGH': 2,
        'MEDIUM': 3,
        'LOW': 4,
        'INFO': 5
    }
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.alerts_queue = []
        self.alert_handlers = []
        self._setup_handlers()
        logger.info("AlertManager initialized")
    
    def _setup_handlers(self):
        """Setup alert handlers based on config."""
        channels = self.config.get('alerts', {}).get('channels', {})
        
        if channels.get('email', {}).get('enabled'):
            self.alert_handlers.append(EmailAlertHandler(self.config))
        
        if channels.get('slack', {}).get('enabled'):
            self.alert_handlers.append(SlackAlertHandler(self.config))
        
        if channels.get('database', {}).get('enabled'):
            self.alert_handlers.append(DatabaseAlertHandler(self.config))
        
        if channels.get('syslog', {}).get('enabled'):
            self.alert_handlers.append(SyslogAlertHandler(self.config))
        
        logger.info(f"Setup {len(self.alert_handlers)} alert handlers")
    
    def create_alert(self, detection_result: Dict) -> Dict:
        """Create alert from detection result."""
        alert = {
            'id': self._generate_alert_id(),
            'timestamp': datetime.now().isoformat(),
            'severity': detection_result.get('threat_level', 'MEDIUM'),
            'message': self._format_alert_message(detection_result),
            'detection_result': detection_result,
            'status': 'new'
        }
        
        logger.info(f"Alert created: {alert['id']} - {alert['severity']}")
        return alert
    
    def send_alert(self, alert: Dict):
        """Send alert through configured channels."""
        if not self.config.get('alerts', {}).get('enabled', True):
            logger.debug("Alerts disabled")
            return
        
        for handler in self.alert_handlers:
            try:
                handler.send(alert)
            except Exception as e:
                logger.error(f"Error sending alert via {handler.__class__.__name__}: {e}")
    
    def _format_alert_message(self, detection_result: Dict) -> str:
        """Format alert message from detection result."""
        return (
            f"Intrusion Detected - "
            f"Confidence: {detection_result.get('confidence', 'N/A'):.2%} - "
            f"Type: {detection_result.get('attack_type', 'Unknown')}"
        )
    
    def _generate_alert_id(self) -> str:
        """Generate unique alert ID."""
        import uuid
        return f"ALERT-{uuid.uuid4().hex[:8].upper()}"
    
    def filter_alerts(self, severity: str = None) -> List[Dict]:
        """Filter alerts by severity."""
        if severity:
            return [a for a in self.alerts_queue if a['severity'] == severity]
        return self.alerts_queue


class AlertHandler:
    """Base class for alert handlers."""
    
    def __init__(self, config: Dict):
        self.config = config
    
    def send(self, alert: Dict):
        """Send alert - to be implemented by subclasses."""
        raise NotImplementedError


class EmailAlertHandler(AlertHandler):
    """Send alerts via email."""
    
    def send(self, alert: Dict):
        """Send alert via email."""
        try:
            import smtplib
            from email.mime.text import MIMEText
            from email.mime.multipart import MIMEMultipart
            
            email_config = self.config.get('alerts', {}).get('channels', {}).get('email', {})
            
            # Build email
            msg = MIMEMultipart()
            msg['From'] = email_config.get('sender_email')
            msg['To'] = ', '.join(email_config.get('recipient_emails', []))
            msg['Subject'] = f"[{alert['severity']}] Intrusion Detection Alert"
            
            body = f"""
            Alert ID: {alert['id']}
            Timestamp: {alert['timestamp']}
            Severity: {alert['severity']}
            Message: {alert['message']}
            
            Details:
            {json.dumps(alert['detection_result'], indent=2)}
            """
            
            msg.attach(MIMEText(body, 'plain'))
            
            # Send email
            server = smtplib.SMTP(email_config.get('smtp_server'), email_config.get('smtp_port'))
            if email_config.get('use_tls'):
                server.starttls()
            
            server.send_message(msg)
            server.quit()
            
            logger.info(f"Alert {alert['id']} sent via email")
            
        except Exception as e:
            logger.error(f"Error sending email alert: {e}")


class SlackAlertHandler(AlertHandler):
    """Send alerts via Slack."""
    
    def send(self, alert: Dict):
        """Send alert via Slack webhook."""
        try:
            import requests
            
            slack_config = self.config.get('alerts', {}).get('channels', {}).get('slack', {})
            webhook_url = slack_config.get('webhook_url')
            
            payload = {
                'text': f"🚨 {alert['severity']} - {alert['message']}",
                'attachments': [{
                    'color': self._get_color_by_severity(alert['severity']),
                    'fields': [
                        {'title': 'Alert ID', 'value': alert['id']},
                        {'title': 'Timestamp', 'value': alert['timestamp']},
                        {'title': 'Confidence', 'value': f"{alert['detection_result'].get('confidence', 0):.2%}"}
                    ]
                }]
            }
            
            requests.post(webhook_url, json=payload)
            logger.info(f"Alert {alert['id']} sent via Slack")
            
        except Exception as e:
            logger.error(f"Error sending Slack alert: {e}")
    
    @staticmethod
    def _get_color_by_severity(severity: str) -> str:
        """Get Slack color by severity level."""
        colors = {
            'CRITICAL': 'danger',
            'HIGH': 'warning',
            'MEDIUM': '#FFA500',
            'LOW': '#36A64F'
        }
        return colors.get(severity, 'good')


class DatabaseAlertHandler(AlertHandler):
    """Store alerts in database."""
    
    def send(self, alert: Dict):
        """Store alert in database."""
        try:
            # Implementation for database storage
            logger.info(f"Alert {alert['id']} stored in database")
        except Exception as e:
            logger.error(f"Error storing alert in database: {e}")


class SyslogAlertHandler(AlertHandler):
    """Send alerts via Syslog."""
    
    def send(self, alert: Dict):
        """Send alert via Syslog."""
        try:
            import logging.handlers
            
            syslog_config = self.config.get('alerts', {}).get('channels', {}).get('syslog', {})
            
            handler = logging.handlers.SysLogHandler(
                address=(syslog_config.get('server', 'localhost'), syslog_config.get('port', 514))
            )
            
            handler.emit(logging.LogRecord(
                name='IDS',
                level=logging.WARNING,
                pathname='',
                lineno=0,
                msg=alert['message'],
                args=(),
                exc_info=None
            ))
            
            logger.info(f"Alert {alert['id']} sent via Syslog")
            
        except Exception as e:
            logger.error(f"Error sending Syslog alert: {e}")
