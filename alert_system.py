"""
Alert System & Risk Logging for AI4HSSE
Monitors REBA scores and triggers alerts for sustained high-risk postures
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import json
from collections import deque

class RiskAlertSystem:
    def __init__(self, high_risk_threshold=7, alert_duration=5.0):
        """
        Initialize alert system
        
        Args:
            high_risk_threshold: REBA score above which is considered high risk
            alert_duration: Seconds of sustained high risk before alert
        """
        self.high_risk_threshold = high_risk_threshold
        self.alert_duration = alert_duration
        
        # Tracking variables
        self.risk_history = deque(maxlen=150)  # Last ~5 seconds at 30 FPS
        self.alert_active = False
        self.alert_start_time = None
        self.current_incident = None
        
        # Incident logging
        self.incidents = []
        self.session_start = datetime.now()
        
        # Statistics
        self.total_frames = 0
        self.high_risk_frames = 0
        self.alert_count = 0
        # Peak logging controls
        self.log_peak_observations = True
        self.last_peak_logged_time = None

        # Create logs directory
        os.makedirs('logs', exist_ok=True)
    
    def update(self, reba_score, risk_level, angles):
        """
        Update alert system with new REBA score
        
        Args:
            reba_score: Current REBA score
            risk_level: Current risk level (low/medium/high/very_high)
            angles: Dictionary of joint angles
            
        Returns:
            dict: Alert status and information
        """
        current_time = datetime.now()
        self.total_frames += 1
        
        # Add to history
        self.risk_history.append({
            'time': current_time,
            'reba_score': reba_score,
            'risk_level': risk_level,
            'angles': angles
        })
        
        # Check if currently in high risk
        is_high_risk = reba_score >= self.high_risk_threshold
        
        if is_high_risk:
            self.high_risk_frames += 1
        
        # Calculate sustained high risk duration
        sustained_duration = self._calculate_sustained_high_risk()
        
        alert_info = {
            'alert_active': False,
            'alert_type': None,
            'message': '',
            'sustained_duration': sustained_duration,
            'should_trigger': False
        }
        
        # Check if alert should be triggered
        if sustained_duration >= self.alert_duration:
            if not self.alert_active:
                # NEW ALERT - start incident
                self.alert_active = True
                self.alert_start_time = current_time
                self.alert_count += 1
                
                self.current_incident = {
                    'incident_id': self.alert_count,
                    'start_time': current_time,
                    'end_time': None,
                    'peak_reba': reba_score,
                    'duration': 0,
                    'risk_level': risk_level,
                    'angles': angles.copy() if angles else {}
                }
                
                alert_info['alert_active'] = True
                alert_info['alert_type'] = 'NEW'
                alert_info['message'] = f'⚠️ HIGH RISK ALERT! REBA Score: {reba_score}'
                alert_info['should_trigger'] = True
                
                print(f"🚨 ALERT TRIGGERED! REBA={reba_score}, Duration={sustained_duration:.2f}s")
                
            else:
                # ONGOING ALERT - update incident
                if self.current_incident:
                    self.current_incident['duration'] = (current_time - self.current_incident['start_time']).total_seconds()
                    if reba_score > self.current_incident['peak_reba']:
                        self.current_incident['peak_reba'] = reba_score
                
                alert_info['alert_active'] = True
                alert_info['alert_type'] = 'ONGOING'
                alert_info['message'] = f'⚠️ Still in high risk! Duration: {sustained_duration:.1f}s'
        
        else:
            # No sustained high risk
            if self.alert_active:
                # Alert just ended - log incident
                if self.current_incident:
                    self.current_incident['end_time'] = current_time
                    self.current_incident['duration'] = (current_time - self.current_incident['start_time']).total_seconds()
                    self.incidents.append(self.current_incident)
                    
                    # Save incident to log
                    self._save_incident(self.current_incident)
                    
                    print(f"✅ Alert cleared. Incident logged: Duration={self.current_incident['duration']:.2f}s")
                
                self.alert_active = False
                self.alert_start_time = None
                self.current_incident = None
                
                alert_info['alert_type'] = 'CLEARED'
                alert_info['message'] = '✅ Risk cleared - posture improved'
        
        # Optionally log brief very-high observations (non-sustained spikes)
        try:
            if getattr(self, 'log_peak_observations', False) and reba_score >= 11:
                should_log_peak = False
                # Avoid logging duplicates too quickly
                if not self.incidents:
                    should_log_peak = True
                else:
                    if self.last_peak_logged_time is None:
                        should_log_peak = True
                    else:
                        if (current_time - self.last_peak_logged_time).total_seconds() > 5:
                            should_log_peak = True

                if should_log_peak:
                    # Create a short incident record for the peak
                    self.alert_count += 1
                    peak_incident = {
                        'incident_id': self.alert_count,
                        'start_time': current_time,
                        'end_time': current_time,
                        'peak_reba': reba_score,
                        'duration': 0.1,
                        'risk_level': risk_level,
                        'angles': angles.copy() if angles else {}
                    }
                    self.incidents.append(peak_incident)
                    self._save_incident(peak_incident)
                    self.last_peak_logged_time = current_time
                    alert_info['message'] = (alert_info.get('message', '') + f' [Logged peak REBA {reba_score}]').strip()
                    alert_info['logged_peak'] = True
        except Exception:
            pass

        return alert_info
    
    def _calculate_sustained_high_risk(self):
        """Calculate how long high risk has been sustained"""
        if len(self.risk_history) < 2:
            return 0
        
        # Count consecutive high risk frames from the end
        consecutive_high_risk = 0
        
        for item in reversed(self.risk_history):
            if item['reba_score'] >= self.high_risk_threshold:
                consecutive_high_risk += 1
            else:
                break
        
        if consecutive_high_risk == 0:
            return 0
        
        # Calculate time span of consecutive high risk frames
        # Get the timestamps of the first and last high-risk frames in the sequence
        high_risk_frames = list(self.risk_history)[-consecutive_high_risk:]
        
        if len(high_risk_frames) < 2:
            # Only one frame, return minimal time
            return 0.1
        
        time_span = (high_risk_frames[-1]['time'] - high_risk_frames[0]['time']).total_seconds()
        
        # Add a small buffer for the current frame (assuming ~30 FPS = 0.033s per frame)
        time_span += 0.033
        
        return time_span
    
    def _save_incident(self, incident):
        """Save incident to log file"""
        log_file = f"logs/incidents_{self.session_start.strftime('%Y%m%d')}.csv"
        
        incident_data = {
            'incident_id': incident['incident_id'],
            'date': incident['start_time'].strftime('%Y-%m-%d'),
            'start_time': incident['start_time'].strftime('%H:%M:%S'),
            'end_time': incident['end_time'].strftime('%H:%M:%S') if incident['end_time'] else '',
            'duration_seconds': round(incident['duration'], 2),
            'peak_reba_score': incident['peak_reba'],
            'risk_level': incident['risk_level'],
            'trunk_angle': incident['angles'].get('trunk_flexion', 'N/A'),
            'neck_angle': incident['angles'].get('neck_flexion', 'N/A')
        }
        
        # Append to CSV
        df = pd.DataFrame([incident_data])
        
        if os.path.exists(log_file):
            df.to_csv(log_file, mode='a', header=False, index=False)
        else:
            df.to_csv(log_file, mode='w', header=True, index=False)
    
    def get_session_summary(self):
        """Get summary statistics for current session"""
        session_duration = (datetime.now() - self.session_start).total_seconds()
        
        high_risk_percentage = (self.high_risk_frames / self.total_frames * 100) if self.total_frames > 0 else 0
        
        total_incident_duration = sum(inc['duration'] for inc in self.incidents)
        # Observed risk counts from recent frame history (helps show very_high even if not logged)
        observed_counts = {'low': 0, 'medium': 0, 'high': 0, 'very_high': 0}
        max_observed_reba = 0
        for h in list(self.risk_history):
            rl = h.get('risk_level')
            if rl in observed_counts:
                observed_counts[rl] += 1
            try:
                max_observed_reba = max(max_observed_reba, float(h.get('reba_score', 0)))
            except Exception:
                pass

        summary = {
            'session_start': self.session_start.strftime('%Y-%m-%d %H:%M:%S'),
            'session_duration': round(session_duration, 1),
            'total_frames_analyzed': self.total_frames,
            'high_risk_frames': self.high_risk_frames,
            'high_risk_percentage': round(high_risk_percentage, 1),
            'total_alerts': self.alert_count,
            'total_incidents_logged': len(self.incidents),
            'total_incident_duration': round(total_incident_duration, 1),
            'average_incident_duration': round(total_incident_duration / len(self.incidents), 1) if self.incidents else 0
        }

        # Add observed frame-level stats to summary for diagnostic / UI display
        summary['observed_risk_counts'] = observed_counts
        summary['max_observed_reba'] = max_observed_reba
        
        return summary
    
    def get_recent_incidents(self, limit=10):
        """Get most recent incidents"""
        return self.incidents[-limit:] if self.incidents else []
    
    def generate_daily_report(self, date=None):
        """Generate daily incident report"""
        if date is None:
            date = datetime.now().strftime('%Y%m%d')
        
        log_file = f"logs/incidents_{date}.csv"
        
        if not os.path.exists(log_file):
            return None
        
        df = pd.read_csv(log_file)
        
        report = {
            'date': date,
            'total_incidents': len(df),
            'total_exposure_time': df['duration_seconds'].sum(),
            'average_reba_score': df['peak_reba_score'].mean(),
            'max_reba_score': df['peak_reba_score'].max(),
            'incidents_by_risk': df['risk_level'].value_counts().to_dict(),
            'longest_incident': df['duration_seconds'].max(),
            'incidents_per_hour': len(df) / (df['duration_seconds'].sum() / 3600) if df['duration_seconds'].sum() > 0 else 0
        }
        
        return report
    
    def export_session_report(self, filename=None):
        """Export detailed session report"""
        if filename is None:
            filename = f"logs/session_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        summary = self.get_session_summary()
        
        report = {
            'summary': summary,
            'incidents': [
                {
                    'incident_id': inc['incident_id'],
                    'start_time': inc['start_time'].strftime('%Y-%m-%d %H:%M:%S'),
                    'duration': round(inc['duration'], 2),
                    'peak_reba': inc['peak_reba'],
                    'risk_level': inc['risk_level']
                }
                for inc in self.incidents
            ]
        }
        
        with open(filename, 'w') as f:
            json.dump(report, f, indent=2)
        
        return filename

# Alert sound generator (optional - for audio alerts)
def play_alert_sound():
    """Play alert sound (platform dependent)"""
    try:
        import winsound
        # Play beep on Windows
        winsound.Beep(1000, 500)  # 1000 Hz for 500ms
    except:
        # Platform doesn't support winsound
        try:
            # Try printing bell character (terminal beep)
            print('\a')
        except:
            pass

# Visualization helper
def create_alert_overlay(frame, alert_info, reba_score):
    """Create visual alert overlay on frame"""
    import cv2
    
    if alert_info['alert_active']:
        # Add red flashing border
        if alert_info['alert_type'] == 'NEW' or (datetime.now().microsecond // 100000) % 2 == 0:
            # Flash effect
            border_color = (0, 0, 255)  # Red
            thickness = 15
            
            h, w = frame.shape[:2]
            cv2.rectangle(frame, (0, 0), (w, h), border_color, thickness)
        
        # Add alert text
        alert_text = alert_info['message']
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.2
        thickness = 3
        
        # Get text size for background
        (text_width, text_height), _ = cv2.getTextSize(alert_text, font, font_scale, thickness)
        
        # Draw background rectangle
        cv2.rectangle(frame, (10, 10), (text_width + 20, text_height + 30), (0, 0, 255), -1)
        
        # Draw text
        cv2.putText(frame, alert_text, (15, text_height + 20), font, font_scale, (255, 255, 255), thickness)
    
    elif alert_info['alert_type'] == 'CLEARED':
        # Show cleared message briefly
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, alert_info['message'], (15, 50), font, 1.0, (0, 255, 0), 2)
    
    return frame