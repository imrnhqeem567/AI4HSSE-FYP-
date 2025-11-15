"""Utility functions for the AI4HSSE system"""

import cv2
import numpy as np
from config import RISK_LEVELS

def resize_image(image, max_width=800):
    """Resize image while maintaining aspect ratio"""
    height, width = image.shape[:2]
    if width > max_width:
        ratio = max_width / width
        new_width = int(width * ratio)
        new_height = int(height * ratio)
        image = cv2.resize(image, (new_width, new_height))
    return image

def add_text_overlay(image, text, position, color=(255, 255, 255), font_scale=0.8):
    """Add text overlay to image"""
    font = cv2.FONT_HERSHEY_SIMPLEX
    thickness = 2
    
    # Add background rectangle for better readability
    (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, thickness)
    cv2.rectangle(image, (position[0] - 5, position[1] - text_height - 10),
                 (position[0] + text_width + 5, position[1] + 5), (0, 0, 0), -1)
    
    cv2.putText(image, text, position, font, font_scale, color, thickness)
    return image

def create_risk_visualization(image, reba_score, risk_level):
    """Add risk level visualization to image"""
    risk_info = RISK_LEVELS[risk_level]
    color_map = {
        'green': (0, 255, 0),
        'yellow': (0, 255, 255),
        'orange': (0, 165, 255),
        'red': (0, 0, 255)
    }
    
    color = color_map[risk_info['color']]
    
    # Add colored border
    border_thickness = 10
    height, width = image.shape[:2]
    cv2.rectangle(image, (0, 0), (width, height), color, border_thickness)
    
    # Add risk information
    text_lines = [
        f"REBA Score: {reba_score}",
        f"Risk Level: {risk_level.upper()}",
        f"Action: {risk_info['action']}"
    ]
    
    y_position = 30
    for line in text_lines:
        add_text_overlay(image, line, (20, y_position), color)
        y_position += 35
    
    return image

def save_analysis_report(results, filename):
    """Save analysis results to file"""
    import json
    with open(filename, 'w') as f:
        json.dump(results, f, indent=2)

def generate_posture_feedback(angles, reba_details):
    """
    Generate specific posture correction advice based on joint angles and REBA scores
    
    Args:
        angles: Dictionary of joint angles
        reba_details: Dictionary of REBA component scores
        
    Returns:
        List of feedback strings
    """
    feedback = []
    
    # Check trunk posture
    trunk_score = reba_details.get('trunk_score', 1)
    trunk_angle = angles.get('trunk_flexion', 0)
    
    if trunk_score >= 4:
        feedback.append(f"⚠️ Back severely bent ({trunk_angle:.0f}°)")
        feedback.append("💡 Straighten back immediately")
    elif trunk_score >= 3:
        feedback.append(f"⚠️ Back bent {trunk_angle:.0f}° - too much")
        feedback.append("💡 Bend knees, not your back")
    elif trunk_score >= 2:
        feedback.append(f"⚠️ Slight forward lean ({trunk_angle:.0f}°)")
        feedback.append("💡 Stand more upright")
    
    # Check neck posture
    neck_score = reba_details.get('neck_score', 1)
    neck_angle = angles.get('neck_flexion', 0)
    
    if neck_score >= 2:
        feedback.append(f"⚠️ Neck bent forward ({neck_angle:.0f}°)")
        feedback.append("💡 Look straight ahead")
    
    # Check upper arm posture
    upper_arm_score = reba_details.get('upper_arm_score', 1)
    upper_arm_angle = reba_details.get('upper_arm_angle', 0)
    
    if upper_arm_score >= 4:
        feedback.append("⚠️ Arms raised very high")
        feedback.append("💡 Use platform or ladder")
    elif upper_arm_score >= 3:
        feedback.append("⚠️ Arms raised above shoulder")
        feedback.append("💡 Lower work height")
    elif upper_arm_score >= 2:
        feedback.append("⚠️ Arms extended forward")
        feedback.append("💡 Work closer to body")
    
    # If everything is good
    if not feedback:
        feedback.append("✅ Excellent posture!")
        feedback.append("✅ Keep it up!")
    
    return feedback