#!/usr/bin/env python3
"""
Automated Response Handler
Executes automated responses to detected intrusions
"""

import logging
from typing import Dict, List
import subprocess
import json
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class ResponseHandler:
    """Handles automated responses to intrusion detections."""
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.response_log = []
        self.blocked_ips = set()
        logger.info("ResponseHandler initialized")
    
    def handle_alert(self, alert: Dict):
        """Execute response actions based on alert."""
        if not self.config.get('response', {}).get('enabled', True):
            logger.info("Response handling disabled")
            return
        
        threat_level = alert.get('severity')
        
        logger.info(f"Handling alert with severity: {threat_level}")
        
        # Get configured response actions
        actions = self.config.get('response', {}).get('response_actions', [])
        
        for action in actions:
            if action.get('enabled', False):
                action_type = action.get('type')
                
                if action_type == 'log':
                    self._action_log(alert)
                elif action_type == 'email':
                    self._action_email(alert)
                elif action_type == 'block_ip':
                    duration = action.get('duration', 3600)
                    self._action_block_ip(alert, duration)
                elif action_type == 'close_connection':
                    self._action_close_connection(alert)
                elif action_type == 'snapshot':
                    save_path = action.get('save_path', 'logs/snapshots/')
                    self._action_snapshot(alert, save_path)
    
    def _action_log(self, alert: Dict):
        """Log the alert to file."""
        try:
            log_entry = {
                'timestamp': datetime.now().isoformat(),
                'alert_id': alert.get('id'),
                'severity': alert.get('severity'),
                'message': alert.get('message')
            }
            
            self.response_log.append(log_entry)
            logger.info(f"Alert logged: {alert.get('id')}")
            
        except Exception as e:
            logger.error(f"Error logging alert: {e}")
    
    def _action_email(self, alert: Dict):
        """Send email notification."""
        try:
            logger.info(f"Sending email for alert: {alert.get('id')}")
            # Email sending is handled by AlertManager
            
        except Exception as e:
            logger.error(f"Error sending email: {e}")
    
    def _action_block_ip(self, alert: Dict, duration: int = 3600):
        """Block IP address."""
        try:
            source_ip = alert.get('detection_result', {}).get('source_ip')
            
            if not source_ip:
                logger.warning("No source IP found in alert")
                return
            
            # Add to blocked IPs set
            self.blocked_ips.add(source_ip)
            
            # Record block action
            unblock_time = datetime.now() + timedelta(seconds=duration)
            
            logger.info(f"Blocked IP: {source_ip} until {unblock_time.isoformat()}")
            
            # Execute iptables rule (Linux-specific)
            # Uncomment to actually block IPs
            # subprocess.run(['sudo', 'iptables', '-A', 'INPUT', '-s', source_ip, '-j', 'DROP'])
            
        except Exception as e:
            logger.error(f"Error blocking IP: {e}")
    
    def _action_close_connection(self, alert: Dict):
        """Close detected malicious connection."""
        try:
            source_ip = alert.get('detection_result', {}).get('source_ip')
            dest_port = alert.get('detection_result', {}).get('dest_port')
            
            if source_ip and dest_port:
                logger.info(f"Closing connection: {source_ip}:{dest_port}")
                # Kill specific connection using tcpkill or similar
                # subprocess.run(['tcpkill', '-i', 'eth0', 'host', source_ip])
            
        except Exception as e:
            logger.error(f"Error closing connection: {e}")
    
    def _action_snapshot(self, alert: Dict, save_path: str):
        """Take network snapshot for forensics."""
        try:
            import os
            os.makedirs(save_path, exist_ok=True)
            
            filename = f"{save_path}/snapshot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            
            with open(filename, 'w') as f:
                json.dump(alert, f, indent=2)
            
            logger.info(f"Network snapshot saved to {filename}")
            
        except Exception as e:
            logger.error(f"Error taking snapshot: {e}")
    
    def get_blocked_ips(self) -> List[str]:
        """Get list of currently blocked IPs."""
        return list(self.blocked_ips)
    
    def unblock_ip(self, ip: str):
        """Unblock an IP address."""
        try:
            if ip in self.blocked_ips:
                self.blocked_ips.remove(ip)
                logger.info(f"Unblocked IP: {ip}")
                # Remove iptables rule
                # subprocess.run(['sudo', 'iptables', '-D', 'INPUT', '-s', ip, '-j', 'DROP'])
            else:
                logger.warning(f"IP {ip} not in blocked list")
        
        except Exception as e:
            logger.error(f"Error unblocking IP: {e}")
    
    def get_response_log(self) -> List[Dict]:
        """Get log of executed responses."""
        return self.response_log
