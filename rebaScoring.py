"""REBA scoring implementation"""

from config import REBA_THRESHOLDS, RISK_LEVELS

class REBAScorer:
    def __init__(self):
        self.trunk_table = self._create_trunk_table()
        self.neck_table = self._create_neck_table()
        self.leg_table = self._create_leg_table()
        self.upper_arm_table = self._create_upper_arm_table()
        self.table_a = self._create_table_a()
        self.table_b = self._create_table_b()
        self.table_c = self._create_table_c()
    
    def _create_trunk_table(self):
        """Create trunk scoring table based on REBA standards"""
        return {
            'upright': 1,
            'slight_flex': 2,
            'flex': 3,
            'severe_flex': 4
        }
    
    def _create_neck_table(self):
        """Create neck scoring table"""
        return {
            'neutral': 1,
            'flex': 2,
            'extend': 2
        }
    
    def _create_leg_table(self):
        """Create leg scoring table (simplified)"""
        return {
            'bilateral_support': 1,
            'unilateral_support': 2
        }
    
    def _create_upper_arm_table(self):
        """Create upper arm scoring table"""
        return {
            'neutral': 1,
            'flex_20': 2,
            'flex_45': 3,
            'flex_90': 4
        }
    
    def _create_table_a(self):
        """REBA Table A: Trunk, Neck, Legs combination"""
        # Simplified version - in real implementation, use official REBA tables
        return {
            (1, 1, 1): 1, (1, 1, 2): 2, (1, 2, 1): 2, (1, 2, 2): 3,
            (2, 1, 1): 2, (2, 1, 2): 3, (2, 2, 1): 3, (2, 2, 2): 4,
            (3, 1, 1): 3, (3, 1, 2): 4, (3, 2, 1): 4, (3, 2, 2): 5,
            (4, 1, 1): 4, (4, 1, 2): 5, (4, 2, 1): 5, (4, 2, 2): 6
        }
    
    def _create_table_b(self):
        """REBA Table B: Upper arms, Lower arms, Wrists"""
        # Simplified - implement full table based on REBA standard
        return {
            (1, 1, 1): 1, (1, 1, 2): 2, (1, 2, 1): 2, (1, 2, 2): 2,
            (2, 1, 1): 1, (2, 1, 2): 2, (2, 2, 1): 2, (2, 2, 2): 3,
            (3, 1, 1): 3, (3, 1, 2): 3, (3, 2, 1): 3, (3, 2, 2): 4,
            (4, 1, 1): 4, (4, 1, 2): 4, (4, 2, 1): 4, (4, 2, 2): 5
        }
    
    def _create_table_c(self):
        """REBA Table C: Final score combination"""
        # Simplified final combination table
        return {
            (1, 1): 1, (1, 2): 1, (1, 3): 2, (1, 4): 3, (1, 5): 4,
            (2, 1): 1, (2, 2): 2, (2, 3): 3, (2, 4): 4, (2, 5): 5,
            (3, 1): 2, (3, 2): 3, (3, 3): 4, (3, 4): 5, (3, 5): 6,
            (4, 1): 3, (4, 2): 4, (4, 3): 5, (4, 4): 6, (4, 5): 7,
            (5, 1): 4, (5, 2): 5, (5, 3): 6, (5, 4): 7, (5, 5): 8,
            (6, 1): 6, (6, 2): 7, (6, 3): 8, (6, 4): 9, (6, 5): 10
        }
    
    def get_trunk_score(self, trunk_angle):
        """Get trunk score based on flexion angle - IMPROVED THRESHOLDS"""
        if trunk_angle is None:
            return 1
        
        # More realistic thresholds for trunk
        # 0-5 degrees = essentially upright
        if trunk_angle <= 5:
            return 1
        elif trunk_angle <= 20:
            return 2
        elif trunk_angle <= 60:
            return 3
        else:
            return 4
    
    def get_neck_score(self, neck_angle):
        """Get neck score based on flexion angle"""
        if neck_angle is None:
            return 1
        
        if neck_angle <= 20:
            return 1
        else:
            return 2
    
    def get_upper_arm_score(self, upper_arm_angle):
        """Get upper arm score"""
        if upper_arm_angle is None:
            return 1
        
        abs_angle = abs(upper_arm_angle)
        
        if abs_angle <= 20:
            return 1
        elif abs_angle <= 45:
            return 2
        elif abs_angle <= 90:
            return 3
        else:
            return 4
    
    def calculate_reba_score(self, angles, load_weight=0, coupling='good'):
        """Calculate final REBA score"""
        if angles is None:
            return {'score': 1, 'risk_level': 'low', 'details': {}}
        
        # Calculate individual scores
        trunk_score = self.get_trunk_score(angles.get('trunk_flexion'))
        neck_score = self.get_neck_score(angles.get('neck_flexion'))
        leg_score = 1  # Simplified - assume bilateral support
        
        # Upper arm score (take maximum of left/right)
        left_arm = self.get_upper_arm_score(angles.get('left_upper_arm'))
        right_arm = self.get_upper_arm_score(angles.get('right_upper_arm'))
        upper_arm_score = max(left_arm, right_arm)
        
        lower_arm_score = 1  # Simplified
        wrist_score = 1      # Simplified
        
        # Look up table scores
        score_a = self.table_a.get((trunk_score, neck_score, leg_score), 3)
        score_b = self.table_b.get((upper_arm_score, lower_arm_score, wrist_score), 3)
        
        # Final REBA score
        final_score = self.table_c.get((score_a, score_b), 5)
        
        # Apply adjustments
        # Tiered load effect: more realistic incremental penalty for heavier loads
        # - >=5 kg: +1
        # - >=10 kg: +2
        # - >=20 kg: +3
        if load_weight >= 20:
            final_score += 3
        elif load_weight >= 10:
            final_score += 2
        elif load_weight >= 5:
            final_score += 1

        # Coupling quality penalty: poor coupling increases final score by 1
        if coupling == 'poor':
            final_score += 1
        
        # Determine risk level
        # Cap final score to REBA maximum (15) and determine risk level
        final_score = min(final_score, 15)
        risk_level = self.get_risk_level(final_score)
        
        details = {
            'trunk_score': trunk_score,
            'neck_score': neck_score,
            'leg_score': leg_score,
            'upper_arm_score': upper_arm_score,
            'score_a': score_a,
            'score_b': score_b,
            'trunk_angle': angles.get('trunk_flexion'),
            'neck_angle': angles.get('neck_flexion'),
            'upper_arm_angle': max(angles.get('left_upper_arm', 0), 
                                  angles.get('right_upper_arm', 0))
        }
        
        return {
            'score': final_score,
            'risk_level': risk_level,
            'details': details
        }
    
    def get_risk_level(self, score):
        """Determine risk level from REBA score"""
        if score <= 3:
            return 'low'
        elif score <= 7:
            return 'medium'
        elif score <= 10:
            return 'high'
        else:
            return 'very_high'
