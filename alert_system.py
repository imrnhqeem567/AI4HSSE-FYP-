"""Risk alert system for high-risk posture detection and incident logging"""

import cv2
import json
import os
from datetime import datetime, timedelta
import pandas as pd

def play_alert_sound():
    """Play alert sound on Windows"""
    try:
        import winsound
        winsound.Beep(1000, 300)
    except Exception:
        pass

def create_alert_overlay(image, alert_info, reba_score):
    """Create visual alert overlay on the image"""
    if not alert_info.get('should_trigger', False):
        return image
    
    h, w = image.shape[:2]
    overlay = image.copy()
    
    # Draw red border for alert
    cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 255), 15)
    
    # Add alert text
    alert_text = "⚠️ HIGH RISK ALERT!"
    cv2.putText(overlay, alert_text, (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 
                1.5, (0, 0, 255), 3)
    
    # Blend
    result = cv2.addWeighted(image, 0.7, overlay, 0.3, 0)
    
    return result

class RiskAlertSystem:
    """System for detecting and logging high-risk postures"""
    
    def __init__(self, high_risk_threshold=7, alert_duration=5.0):
        self.high_risk_threshold = high_risk_threshold
        self.alert_duration = alert_duration
        self.high_risk_start_time = None
        self.current_high_reba = 0
        self.incidents = []
        self.log_peak_observations = True
        self.last_peak_logged_time = None
        self.peak_log_cooldown = 5.0  # seconds
        self.session_start_time = datetime.now()
        self.observed_risk_counts = {'low': 0, 'medium': 0, 'high': 0, 'very_high': 0}
        self.max_observed_reba = 0
        
    def update(self, reba_score, risk_level, angles):
        """Update alert status based on current REBA score"""
        alert_info = {
            'should_trigger': False,
            'is_high_risk': reba_score >= self.high_risk_threshold,
            'sustained_duration': 0,
            'message': '',
            'logged_peak': False
        }
        
        # Track observed REBA
        self.max_observed_reba = max(self.max_observed_reba, reba_score)
        
        # Track risk distribution
        if risk_level in self.observed_risk_counts:
            self.observed_risk_counts[risk_level] += 1
        
        # Check if entering high-risk state
        if reba_score >= self.high_risk_threshold:
            if self.high_risk_start_time is None:
                self.high_risk_start_time = datetime.now()
                self.current_high_reba = reba_score
            
            # Calculate sustained duration
            sustained = (datetime.now() - self.high_risk_start_time).total_seconds()
            alert_info['sustained_duration'] = sustained
            
            # Check if alert duration reached
            if sustained >= self.alert_duration:
                alert_info['should_trigger'] = True
                alert_info['message'] = f"High risk sustained for {sustained:.1f}s"
            else:
                alert_info['message'] = f"High risk for {sustained:.1f}s (need {self.alert_duration}s)"
            
            # Update peak REBA
            if reba_score > self.current_high_reba:
                self.current_high_reba = reba_score
            
            # Log peak observations (REBA >= 11) with deduplication
            if self.log_peak_observations and reba_score >= 11:
                now = datetime.now()
                should_log = (
                    self.last_peak_logged_time is None or
                    (now - self.last_peak_logged_time).total_seconds() >= self.peak_log_cooldown
                )
                if should_log:
                    # Log brief peak incident
                    incident = {
                        'incident_id': len(self.incidents) + 1,
                        'start_time': now,
                        'end_time': now,
                        'duration': 0.0,
                        'peak_reba': reba_score,
                        'risk_level': risk_level,
                        'trunk_angle': angles.get('trunk_flexion', 0),
                        'neck_angle': angles.get('neck_flexion', 0)
                    }
                    self.incidents.append(incident)
                    alert_info['logged_peak'] = True
                    self.last_peak_logged_time = now
                    alert_info['message'] = (alert_info.get('message', '') + f' [Logged peak REBA {reba_score}]').strip()
        else:
            # Exiting high-risk state
            if self.high_risk_start_time is not None:
                end_time = datetime.now()
                duration = (end_time - self.high_risk_start_time).total_seconds()
                
                # Create incident record
                if duration >= 0.5:  # Log only if sustained for at least 0.5 seconds
                    incident = {
                        'incident_id': len(self.incidents) + 1,
                        'start_time': self.high_risk_start_time,
                        'end_time': end_time,
                        'duration': duration,
                        'peak_reba': self.current_high_reba,
                        'risk_level': risk_level,
                        'trunk_angle': angles.get('trunk_flexion', 0),
                        'neck_angle': angles.get('neck_flexion', 0)
                    }
                    self.incidents.append(incident)
                
                self.high_risk_start_time = None
                self.current_high_reba = 0
        
        return alert_info
    
    def get_session_summary(self):
        """Get summary of the current session"""
        total_incident_duration = sum(inc['duration'] for inc in self.incidents)
        session_duration = (datetime.now() - self.session_start_time).total_seconds()
        
        high_risk_percentage = (total_incident_duration / session_duration * 100) if session_duration > 0 else 0
        
        return {
            'total_alerts': len(self.incidents),
            'total_incidents_logged': len(self.incidents),
            'total_incident_duration': total_incident_duration,
            'high_risk_percentage': high_risk_percentage,
            'session_duration': session_duration,
            'observed_risk_counts': self.observed_risk_counts,
            'max_observed_reba': self.max_observed_reba
        }
    
    def get_recent_incidents(self, limit=10):
        """Get recent incidents"""
        return self.incidents[-limit:]
    
    def export_session_report(self):
        """Export session report to JSON file"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'session_report_{timestamp}.json'
        
        report = {
            'session_start': self.session_start_time.isoformat(),
            'session_end': datetime.now().isoformat(),
            'summary': self.get_session_summary(),
            'incidents': [
                {
                    'incident_id': inc['incident_id'],
                    'start_time': inc['start_time'].isoformat(),
                    'end_time': inc['end_time'].isoformat(),
                    'duration': inc['duration'],
                    'peak_reba': inc['peak_reba'],
                    'risk_level': inc['risk_level'],
                    'trunk_angle': inc['trunk_angle'],
                    'neck_angle': inc['neck_angle']
                }
                for inc in self.incidents
            ]
        }
        
        with open(filename, 'w') as f:
            json.dump(report, f, indent=2)
        
        return filename
    
    def generate_daily_report(self):
        """Generate daily report of incidents"""
        if not self.incidents:
            return None
        
        incidents_df = pd.DataFrame([
            {
                'start_time': inc['start_time'],
                'end_time': inc['end_time'],
                'duration': inc['duration'],
                'peak_reba': inc['peak_reba'],
                'risk_level': inc['risk_level']
            }
            for inc in self.incidents
        ])
        
        report = {
            'date': datetime.now().strftime('%Y-%m-%d'),
            'total_incidents': len(self.incidents),
            'total_exposure_time': incidents_df['duration'].sum(),
            'average_reba_score': incidents_df['peak_reba'].mean(),
            'peak_reba_score': incidents_df['peak_reba'].max(),
            'high_risk_incidents': len(incidents_df[incidents_df['peak_reba'] >= 11])
        }
        
        return report
    
    def save_incidents_to_csv(self, filename=None):
        """Save incidents to CSV file"""
        if not self.incidents:
            return None
        
        if filename is None:
            date_str = datetime.now().strftime('%Y%m%d')
            filename = f'logs/incidents_{date_str}.csv'
        
        # Create logs directory if it doesn't exist
        os.makedirs(os.path.dirname(filename) if os.path.dirname(filename) else '.', exist_ok=True)
        
        # Prepare data
        data = []
        for inc in self.incidents:
            data.append({
                'incident_id': inc['incident_id'],
                'date': inc['start_time'].strftime('%Y-%m-%d'),
                'start_time': inc['start_time'].strftime('%H:%M:%S'),
                'end_time': inc['end_time'].strftime('%H:%M:%S'),
                'duration_seconds': round(inc['duration'], 2),
                'peak_reba_score': inc['peak_reba'],
                'risk_level': inc['risk_level'],
                'trunk_angle': round(inc['trunk_angle'], 2),
                'neck_angle': round(inc['neck_angle'], 2)
            })
        
        df = pd.DataFrame(data)
        df.to_csv(filename, index=False)
        
        return filename
