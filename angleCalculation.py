"""Joint angle calculations for REBA assessment"""

import numpy as np
import math

def calculate_angle(point1, point2, point3):
    """
    Calculate angle between three points
    point2 is the vertex of the angle
    """
    # Convert to numpy arrays
    a = np.array(point1)
    b = np.array(point2)  # vertex
    c = np.array(point3)
    
    # Calculate vectors
    ba = a - b
    bc = c - b
    
    # Calculate angle using dot product
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    
    # Handle numerical errors
    cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
    
    # Convert to degrees
    angle = np.arccos(cosine_angle)
    angle_degrees = np.degrees(angle)
    
    return angle_degrees

def calculate_trunk_angle(landmarks):
    """Calculate trunk flexion angle - IMPROVED"""
    if landmarks is None or len(landmarks) < 33:
        return None
    
    # Use shoulder midpoint and hip midpoint
    left_shoulder = landmarks[11][:2]
    right_shoulder = landmarks[12][:2]
    left_hip = landmarks[23][:2]
    right_hip = landmarks[24][:2]
    
    # Calculate midpoints
    shoulder_mid = np.array([(left_shoulder[0] + right_shoulder[0])/2,
                             (left_shoulder[1] + right_shoulder[1])/2])
    hip_mid = np.array([(left_hip[0] + right_hip[0])/2,
                        (left_hip[1] + right_hip[1])/2])
    
    # Calculate the trunk vector (from hip to shoulder)
    trunk_vector = shoulder_mid - hip_mid
    
    # Vertical reference vector (pointing up in image coordinates)
    # In image coordinates, y increases downward, so vertical up is (0, -1)
    vertical_vector = np.array([0, -1])
    
    # Calculate angle between trunk and vertical
    dot_product = np.dot(trunk_vector, vertical_vector)
    trunk_length = np.linalg.norm(trunk_vector)
    
    if trunk_length == 0:
        return 0
    
    # Angle from vertical (0 = perfectly upright)
    cos_angle = dot_product / trunk_length
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    
    angle_rad = np.arccos(cos_angle)
    angle_deg = np.degrees(angle_rad)
    
    # The angle_deg now represents deviation from vertical
    # 0 degrees = perfectly upright
    # 90 degrees = horizontal (bent 90 degrees forward)
    
    return angle_deg

def calculate_neck_angle(landmarks):
    """Calculate neck flexion angle using shoulder midpoint and nose vector.

    Returns degrees of deviation from vertical (0 = neutral).
    """
    if landmarks is None or len(landmarks) < 33:
        return None

    nose = landmarks[0][:2]
    left_shoulder = landmarks[11][:2]
    right_shoulder = landmarks[12][:2]

    shoulder_mid = np.array([(left_shoulder[0] + right_shoulder[0]) / 2,
                             (left_shoulder[1] + right_shoulder[1]) / 2])

    # Neck vector (from shoulder midpoint to nose)
    neck_vector = np.array(nose) - shoulder_mid

    # Vertical reference vector (up in image coords)
    vertical_vector = np.array([0, -1])

    # Calculate angle between neck vector and vertical
    neck_length = np.linalg.norm(neck_vector)
    if neck_length == 0:
        return 0

    cos_angle = np.dot(neck_vector, vertical_vector) / neck_length
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    angle_rad = np.arccos(cos_angle)
    angle_deg = np.degrees(angle_rad)

    # We want forward flexion as positive; use the angle from vertical
    return angle_deg

def calculate_upper_arm_angle(landmarks, side='left'):
    """Calculate upper arm angle"""
    if landmarks is None or len(landmarks) < 33:
        return None

    if side == 'left':
        shoulder = landmarks[11][:2]
        elbow = landmarks[13][:2]
        hip = landmarks[23][:2]
    else:  # right side
        shoulder = landmarks[12][:2]
        elbow = landmarks[14][:2]
        hip = landmarks[24][:2]

    # Use the hip as the body reference to compute shoulder-relative arm angle
    # Angle at the shoulder between hip->shoulder and elbow->shoulder
    upper_arm_angle = calculate_angle(hip, shoulder, elbow)

    # Return positive absolute angle (magnitude of elevation/abduction)
    return abs(upper_arm_angle)

def extract_all_angles(landmarks):
    """Extract all relevant angles for REBA assessment"""
    angles = {}
    
    try:
        angles['trunk_flexion'] = calculate_trunk_angle(landmarks)
        angles['neck_flexion'] = calculate_neck_angle(landmarks)
        angles['left_upper_arm'] = calculate_upper_arm_angle(landmarks, 'left')
        angles['right_upper_arm'] = calculate_upper_arm_angle(landmarks, 'right')
        
        # Add more angle calculations as needed
        # angles['lower_arm'] = calculate_lower_arm_angle(landmarks)
        # angles['wrist'] = calculate_wrist_angle(landmarks)
        
    except Exception as e:
        print(f"Error calculating angles: {e}")
        return None
    
    return angles
