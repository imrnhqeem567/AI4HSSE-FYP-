"""Configuration for AI4HSSE system"""

# Pose detection configuration
POSE_CONFIG = {
    'min_detection_confidence': 0.5,
    'min_tracking_confidence': 0.5,
    'model_complexity': 1
}

# REBA thresholds
REBA_THRESHOLDS = {
    'low': (1, 3),
    'medium': (4, 7),
    'high': (8, 10),
    'very_high': (11, 15)
}

# Risk levels and associated information
RISK_LEVELS = {
    'low': {
        'color': 'green',
        'action': 'Monitor',
        'description': 'Acceptable risk level'
    },
    'medium': {
        'color': 'yellow',
        'action': 'Investigate',
        'description': 'Further investigation needed'
    },
    'high': {
        'color': 'orange',
        'action': 'Investigate and Correct',
        'description': 'Intervention required soon'
    },
    'very_high': {
        'color': 'red',
        'action': 'Investigate and Implement Control Measures',
        'description': 'Immediate action required'
    }
}

# Alert system configuration
ALERT_CONFIG = {
    'high_risk_threshold': 7,  # REBA score threshold for alerts
    'alert_duration': 5.0,      # Seconds to maintain high risk before alert
    'peak_logging_enabled': True  # Log peaks >= 11 REBA automatically
}

# Logging configuration
LOGGING_CONFIG = {
    'log_directory': 'logs',
    'date_format': '%Y%m%d',
    'time_format': '%H:%M:%S',
    'incident_log_prefix': 'incidents_'
}
