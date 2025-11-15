"""Pose detection using MediaPipe BlazePose"""

import cv2
import mediapipe as mp
import numpy as np
from config import POSE_CONFIG

class PoseDetector:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.mp_draw = mp.solutions.drawing_utils
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=POSE_CONFIG['min_detection_confidence'],
            min_tracking_confidence=POSE_CONFIG['min_tracking_confidence'],
            model_complexity=POSE_CONFIG['model_complexity']
        )
        # HOG person detector as a fallback to find person bounding boxes in static images
        try:
            self.hog = cv2.HOGDescriptor()
            self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        except Exception:
            self.hog = None
        
    def detect_pose(self, image, static_image=False):
        """Detect pose landmarks in an image.

        Args:
            image: BGR image (numpy array)
            static_image: If True, run MediaPipe in static_image_mode to improve single-image detection
        """
        # Convert BGR to RGB
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Provide multi-attempt detection for static images to improve robustness
        self.last_detection_debug = []

        if static_image:
            # Try several strategies for single-image detection
            attempts = []

            # Strategy 1: default static image mode with configured thresholds
            with self.mp_pose.Pose(
                static_image_mode=True,
                min_detection_confidence=POSE_CONFIG['min_detection_confidence'],
                model_complexity=POSE_CONFIG['model_complexity']
            ) as pose:
                res1 = pose.process(rgb_image)
                attempts.append({'method': 'static_default', 'results': res1})
                if res1.pose_landmarks:
                    self.last_detection_debug = attempts
                    return res1

            # Strategy 2: upscale image and retry (helps when person is small)
            try:
                h, w = rgb_image.shape[:2]
                up = cv2.resize(rgb_image, (int(w * 1.5), int(h * 1.5)))
                with self.mp_pose.Pose(static_image_mode=True, min_detection_confidence=max(0.4, POSE_CONFIG['min_detection_confidence'] - 0.2), model_complexity=2) as pose:
                    res2 = pose.process(up)
                    attempts.append({'method': 'upscale+complex2', 'results': res2})
                    if res2.pose_landmarks:
                        self.last_detection_debug = attempts
                        return res2
            except Exception:
                pass

            # Strategy 3: try horizontal flip (sometimes orientation matters)
            try:
                flipped = cv2.flip(rgb_image, 1)
                with self.mp_pose.Pose(static_image_mode=True, min_detection_confidence=max(0.4, POSE_CONFIG['min_detection_confidence'] - 0.2), model_complexity=POSE_CONFIG['model_complexity']) as pose:
                    res3 = pose.process(flipped)
                    attempts.append({'method': 'flipped', 'results': res3})
                    if res3.pose_landmarks:
                        self.last_detection_debug = attempts
                        return res3
            except Exception:
                pass

            # No successful detection; store attempts for debug and return last result
            # Strategy 4: try HOG person detector to get candidate crops and run pose on each crop
            if self.hog is not None:
                try:
                    gray = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
                except Exception:
                    gray = rgb_image

                try:
                    rects, weights = self.hog.detectMultiScale(gray, winStride=(8, 8), padding=(8, 8), scale=1.05)
                    attempts.append({'method': 'hog_count', 'results': None, 'rects': len(rects)})
                    for (x, y, w, h) in rects:
                        try:
                            crop = rgb_image[y:y+h, x:x+w]
                            with self.mp_pose.Pose(static_image_mode=True, min_detection_confidence=max(0.4, POSE_CONFIG['min_detection_confidence'] - 0.2), model_complexity=POSE_CONFIG['model_complexity']) as pose:
                                r = pose.process(crop)
                                # If landmarks found, remap normalized coordinates from crop -> original image
                                if r and r.pose_landmarks:
                                    # remap each landmark
                                    orig_h, orig_w = rgb_image.shape[:2]
                                    for lm in r.pose_landmarks.landmark:
                                        # lm.x, lm.y are normalized relative to the crop; convert to orig normalized
                                        lm.x = (x + lm.x * w) / orig_w
                                        lm.y = (y + lm.y * h) / orig_h
                                    attempts.append({'method': 'hog_crop', 'results': r, 'rect': (x, y, w, h)})
                                    self.last_detection_debug = attempts
                                    return r
                        except Exception:
                            continue
                except Exception:
                    pass

            self.last_detection_debug = attempts
            return attempts[-1]['results'] if attempts else None
        else:
            # Default video/tracking mode (uses self.pose instance)
            results = self.pose.process(rgb_image)
            return results
    
    def extract_landmarks(self, results):
        """Extract normalized landmark coordinates"""
        # Guard against None results (detection may have failed)
        if results is None or not getattr(results, 'pose_landmarks', None):
            return None

        landmarks = []
        for landmark in results.pose_landmarks.landmark:
            # Include visibility if available as a 4th element to help debugging
            vis = getattr(landmark, 'visibility', 0.0)
            landmarks.append([landmark.x, landmark.y, landmark.z, vis])

        return np.array(landmarks)
    
    def draw_landmarks(self, image, results):
        """Draw pose landmarks on image"""
        if results and getattr(results, 'pose_landmarks', None):
            self.mp_draw.draw_landmarks(
                image, 
                results.pose_landmarks, 
                self.mp_pose.POSE_CONNECTIONS
            )
        return image
