"""Main Streamlit application for AI4HSSE system"""

import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io
import pandas as pd
import os

from poseDetection import PoseDetector
from angleCalculation import extract_all_angles
from rebaScoring import REBAScorer
from utils import resize_image, create_risk_visualization
from config import RISK_LEVELS
from alert_system import RiskAlertSystem, create_alert_overlay
import time

# Initialize components
@st.cache_resource
def load_models():
    pose_detector = PoseDetector()
    reba_scorer = REBAScorer()
    return pose_detector, reba_scorer

def run_camera_analysis_with_alerts(pose_detector, reba_scorer, alert_threshold, alert_duration, load_weight=0, coupling_quality='good'):
    """Run real-time camera analysis with adjustable alert thresholds"""
    
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        st.error("❌ Could not access camera. Please check:")
        st.write("- Camera is connected")
        st.write("- No other application is using the camera")
        st.write("- Browser has camera permission")
        return
    
    st.success("✅ Camera started! A new window will open showing your posture analysis.")
    st.info(f"💡 Alert will trigger at REBA ≥{alert_threshold} for {alert_duration}s. Press ESC or Q to stop.")
    
    # Initialize alert system with custom thresholds
    alert_system = RiskAlertSystem(high_risk_threshold=alert_threshold, alert_duration=alert_duration)
    
    window_name = "AI4HSSE - Real-Time Posture Analysis (Press ESC to exit)"
    
    while cap.isOpened():
        ret, frame = cap.read()
        
        if not ret:
            break
        
        # Flip frame for mirror effect
        frame = cv2.flip(frame, 1)
        
        # Process frame
        result, error = process_image(frame, pose_detector, reba_scorer, load_weight, coupling_quality)
        
        if result:
            # Get annotated image with pose and REBA score
            display_frame = result['annotated_image']
            
            # Update alert system
            alert_info = alert_system.update(
                reba_score=result['reba_result']['score'],
                risk_level=result['reba_result']['risk_level'],
                angles=result['angles']
            )
            
            # Add alert overlay
            display_frame = create_alert_overlay(display_frame, alert_info, result['reba_result']['score'])
            
            # Add session statistics
            summary = alert_system.get_session_summary()
            stats_y = 80
            cv2.putText(display_frame, f"Alerts Today: {summary['total_alerts']}", 
                       (10, stats_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(display_frame, f"High Risk Time: {summary['total_incident_duration']:.1f}s", 
                       (10, stats_y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Show alert progress bar
            sustained_duration = alert_info.get('sustained_duration', 0)
            if sustained_duration > 0:
                progress_pct = min(int((sustained_duration / alert_duration) * 100), 100)
                
                # Show progress even if not at threshold yet
                if result['reba_result']['score'] >= alert_threshold:
                    color = (0, 255, 255) if sustained_duration < alert_duration else (0, 255, 0)
                    cv2.putText(display_frame, f"Alert Progress: {progress_pct}% ({sustained_duration:.1f}s/{alert_duration}s)", 
                               (10, stats_y + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            # Debug info - show current REBA and threshold
            debug_y = display_frame.shape[0] - 60
            cv2.putText(display_frame, f"Current REBA: {result['reba_result']['score']} | Threshold: {alert_threshold}", 
                       (10, debug_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(display_frame, f"Sustained: {sustained_duration:.2f}s | Need: {alert_duration}s", 
                       (10, debug_y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Add posture correction guidance on the right side
            from utils import generate_posture_feedback
            feedback_list = generate_posture_feedback(result['angles'], result['reba_result']['details'])
            
            # Draw feedback on right side of frame
            feedback_x = display_frame.shape[1] - 400  # 400 pixels from right edge
            feedback_y = 100
            
            # Draw background box for feedback
            box_height = len(feedback_list) * 35 + 40
            cv2.rectangle(display_frame, 
                         (feedback_x - 10, feedback_y - 30), 
                         (display_frame.shape[1] - 10, feedback_y + box_height), 
                         (50, 50, 50), -1)  # Dark gray background
            
            # Draw title
            cv2.putText(display_frame, "POSTURE GUIDANCE:", 
                       (feedback_x, feedback_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Draw each feedback item
            for i, feedback_text in enumerate(feedback_list):
                y_pos = feedback_y + 35 + (i * 35)
                # Color based on feedback type
                if "✅" in feedback_text:
                    color = (0, 255, 0)  # Green for good
                elif "⚠️" in feedback_text:
                    color = (0, 165, 255)  # Orange for warning
                elif "💡" in feedback_text:
                    color = (255, 255, 0)  # Yellow for tips
                else:
                    color = (255, 255, 255)  # White default
                
                cv2.putText(display_frame, feedback_text, 
                           (feedback_x, y_pos), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            
            # Play sound on new alert
            if alert_info.get('should_trigger', False):
                try:
                    import winsound
                    winsound.Beep(1000, 300)  # Short beep
                except:
                    pass
            
        else:
            display_frame = frame.copy()
            cv2.putText(display_frame, "No person detected - step back to show full body", 
                       (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        
        # Add instructions
        cv2.putText(display_frame, "Press ESC or Q to exit", (10, display_frame.shape[0] - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Display frame
        cv2.imshow(window_name, display_frame)
        
        # Check for exit
        key = cv2.waitKey(1) & 0xFF
        if key == 27 or key == ord('q'):  # ESC or Q
            break
    
    cap.release()
    cv2.destroyAllWindows()
    
    # Show session summary
    summary = alert_system.get_session_summary()
    
    st.success("✅ Camera stopped successfully!")
    
    st.subheader("📊 Session Summary")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Alerts", summary['total_alerts'])
    
    with col2:
        st.metric("Incidents Logged", summary['total_incidents_logged'])
    
    with col3:
        st.metric("High Risk Time", f"{summary['total_incident_duration']:.1f}s")
    
    with col4:
        st.metric("High Risk %", f"{summary['high_risk_percentage']:.1f}%")
    
    # Show recent incidents
    if summary['total_incidents_logged'] > 0:
        st.subheader("⚠️ Recent Incidents")
        
        recent = alert_system.get_recent_incidents(limit=10)
        incidents_df = pd.DataFrame([
            {
                'Incident #': inc['incident_id'],
                'Time': inc['start_time'].strftime('%H:%M:%S'),
                'Duration (s)': round(inc['duration'], 1),
                'Peak REBA': inc['peak_reba'],
                'Risk Level': inc['risk_level']
            }
            for inc in recent
        ])
        
        st.dataframe(incidents_df, use_container_width=True)
        
        # Export report button
        if st.button("📥 Download Session Report"):
            report_file = alert_system.export_session_report()
            with open(report_file, 'r') as f:
                st.download_button(
                    label="Download JSON Report",
                    data=f.read(),
                    file_name=os.path.basename(report_file),
                    mime="application/json"
                )
    
    # Generate daily report if exists
    daily_report = alert_system.generate_daily_report()
    if daily_report:
        st.subheader("📅 Today's Report")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Incidents", daily_report['total_incidents'])
        with col2:
            st.metric("Total Exposure", f"{daily_report['total_exposure_time']:.1f}s")
        with col3:
            st.metric("Avg REBA Score", f"{daily_report['average_reba_score']:.1f}")

def run_camera_analysis_streamlit(pose_detector, reba_scorer):
    """Run real-time camera analysis with OpenCV window and alert system"""
    
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        st.error("❌ Could not access camera. Please check:")
        st.write("- Camera is connected")
        st.write("- No other application is using the camera")
        st.write("- Browser has camera permission")
        return
    
    st.success("✅ Camera started! A new window will open showing your posture analysis.")
    st.info("💡 Press ESC or Q to stop the camera")
    
    # Initialize alert system
    alert_system = RiskAlertSystem(high_risk_threshold=7, alert_duration=5.0)
    
    window_name = "AI4HSSE - Real-Time Posture Analysis (Press ESC to exit)"
    
    while cap.isOpened():
        ret, frame = cap.read()
        
        if not ret:
            break
        
        # Flip frame for mirror effect
        frame = cv2.flip(frame, 1)
        
        # Process frame
        result, error = process_image(frame, pose_detector, reba_scorer)
        
        if result:
            # Get annotated image with pose and REBA score
            display_frame = result['annotated_image']
            
            # Update alert system
            alert_info = alert_system.update(
                reba_score=result['reba_result']['score'],
                risk_level=result['reba_result']['risk_level'],
                angles=result['angles']
            )
            
            # Add alert overlay
            display_frame = create_alert_overlay(display_frame, alert_info, result['reba_result']['score'])
            
            # Add session statistics
            summary = alert_system.get_session_summary()
            stats_y = 80
            cv2.putText(display_frame, f"Alerts Today: {summary['total_alerts']}", 
                       (10, stats_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(display_frame, f"High Risk Time: {summary['total_incident_duration']:.1f}s", 
                       (10, stats_y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Play sound on new alert
            if alert_info['should_trigger']:
                try:
                    import winsound
                    winsound.Beep(1000, 300)  # Short beep
                except:
                    pass
            
        else:
            display_frame = frame.copy()
            cv2.putText(display_frame, "No person detected - step back to show full body", 
                       (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        
        # Add instructions
        cv2.putText(display_frame, "Press ESC or Q to exit", (10, display_frame.shape[0] - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Display frame
        cv2.imshow(window_name, display_frame)
        
        # Check for exit
        key = cv2.waitKey(1) & 0xFF
        if key == 27 or key == ord('q'):  # ESC or Q
            break
    
    cap.release()
    cv2.destroyAllWindows()
    
    # Show session summary
    summary = alert_system.get_session_summary()
    
    st.success("✅ Camera stopped successfully!")
    
    st.subheader("📊 Session Summary")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Alerts", summary['total_alerts'])
    
    with col2:
        st.metric("Incidents Logged", summary['total_incidents_logged'])
    
    with col3:
        st.metric("High Risk Time", f"{summary['total_incident_duration']:.1f}s")
    
    with col4:
        st.metric("High Risk %", f"{summary['high_risk_percentage']:.1f}%")
    
    # Show recent incidents
    if summary['total_incidents_logged'] > 0:
        st.subheader("⚠️ Recent Incidents")
        
        recent = alert_system.get_recent_incidents(limit=10)
        incidents_df = pd.DataFrame([
            {
                'Incident #': inc['incident_id'],
                'Time': inc['start_time'].strftime('%H:%M:%S'),
                'Duration (s)': round(inc['duration'], 1),
                'Peak REBA': inc['peak_reba'],
                'Risk Level': inc['risk_level']
            }
            for inc in recent
        ])
        
        st.dataframe(incidents_df, use_container_width=True)
        
        # Export report button
        if st.button("📥 Download Session Report"):
            report_file = alert_system.export_session_report()
            with open(report_file, 'r') as f:
                st.download_button(
                    label="Download JSON Report",
                    data=f.read(),
                    file_name=os.path.basename(report_file),
                    mime="application/json"
                )
    
    # Generate daily report if exists
    daily_report = alert_system.generate_daily_report()
    if daily_report:
        st.subheader("📅 Today's Report")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Incidents", daily_report['total_incidents'])
        with col2:
            st.metric("Total Exposure", f"{daily_report['total_exposure_time']:.1f}s")
        with col3:
            st.metric("Avg REBA Score", f"{daily_report['average_reba_score']:.1f}")

def run_camera_analysis(camera_placeholder, metrics_placeholder, pose_detector, reba_scorer, alert_threshold=7, alert_duration=5.0, debug=False, camera_width=480, load_weight=0, coupling_quality='good'):
    """Run real-time camera analysis"""
    
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        st.error("❌ Could not access camera. Please check your webcam connection.")
        return
    
    # Session state for stop button
    if 'stop_camera' not in st.session_state:
        st.session_state.stop_camera = False
    
    # Initialize or update alert system in session state
    if 'alert_system' not in st.session_state:
        try:
            st.session_state.alert_system = RiskAlertSystem(high_risk_threshold=alert_threshold, alert_duration=alert_duration)
        except Exception:
            st.session_state.alert_system = None

    alert_system = st.session_state.alert_system

    # Ensure alert system uses latest slider values
    if alert_system:
        alert_system.high_risk_threshold = alert_threshold
        alert_system.alert_duration = alert_duration
        # Ensure peak logging is enabled by default
        try:
            alert_system.log_peak_observations = True
        except Exception:
            pass

    frame_count = 0
    
    while cap.isOpened() and not st.session_state.stop_camera:
        ret, frame = cap.read()
        
        if not ret:
            st.error("Failed to grab frame from camera")
            break
        
        # Process every frame for smooth experience
        frame_count += 1
        
        # Flip frame horizontally for mirror effect
        frame = cv2.flip(frame, 1)
        
        # Process frame
        result, error = process_image(frame, pose_detector, reba_scorer, load_weight=load_weight, coupling_quality=coupling_quality)

        if result:
            # Get annotated image (BGR)
            display_frame = result['annotated_image']

            # If alerting is enabled, update alert system and overlay
            if alert_system:
                alert_info = alert_system.update(
                    reba_score=result['reba_result']['score'],
                    risk_level=result['reba_result']['risk_level'],
                    angles=result['angles']
                )

                # Apply visual overlay (works on BGR frame)
                display_frame = create_alert_overlay(display_frame, alert_info, result['reba_result']['score'])

                # Play sound if a new alert should trigger
                if alert_info.get('should_trigger', False):
                    try:
                        from alert_system import play_alert_sound
                        play_alert_sound()
                    except Exception:
                        pass

            # Convert BGR to RGB for display
            display_frame = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)

            # Display frame with configured width
            camera_placeholder.image(display_frame, channels="RGB", width=camera_width)
            
            # Display metrics
            reba_result = result['reba_result']
            
            with metrics_placeholder.container():
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("REBA Score", reba_result['score'])
                
                with col2:
                    risk_level = reba_result['risk_level'].replace('_', ' ').title()
                    st.metric("Risk Level", risk_level)
                
                with col3:
                    trunk_angle = result['angles'].get('trunk_flexion')
                    if trunk_angle:
                        st.metric("Trunk Angle", f"{trunk_angle:.1f}°")
                
                with col4:
                    neck_angle = result['angles'].get('neck_flexion')
                    if neck_angle:
                        st.metric("Neck Angle", f"{neck_angle:.1f}°")
        else:
            # Show original frame if no person detected
            display_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            camera_placeholder.image(display_frame, channels="RGB", width=camera_width)
            metrics_placeholder.warning("⚠️ No person detected in frame")
        
        # Small delay to prevent overloading
        time.sleep(0.03)  # ~30 FPS
    
    cap.release()
    # Show session summary (Streamlit-friendly)
    try:
        summary = alert_system.get_session_summary() if alert_system else None
    except Exception:
        summary = None

    if summary:
        st.success("✅ Camera stopped successfully!")
        st.subheader("📊 Session Summary")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Alerts", summary.get('total_alerts', 0))
        with col2:
            st.metric("Incidents Logged", summary.get('total_incidents_logged', 0))
        with col3:
            st.metric("High Risk Time", f"{summary.get('total_incident_duration', 0):.1f}s")
        with col4:
            st.metric("High Risk %", f"{summary.get('high_risk_percentage', 0):.1f}%")

        # Observed frame-level stats
        observed = summary.get('observed_risk_counts', {})
        st.markdown("**Observed frame counts (recent history):**")
        st.write(f"Low: {observed.get('low',0)}, Medium: {observed.get('medium',0)}, High: {observed.get('high',0)}, Very High: {observed.get('very_high',0)}")
        st.write(f"Max observed REBA: {summary.get('max_observed_reba', 0)}")
        
        # Save incidents to CSV for incident logs view
        if summary.get('total_incidents_logged', 0) > 0 or (alert_system and len(alert_system.incidents) > 0):
            try:
                csv_file = alert_system.save_incidents_to_csv()
                st.success(f"✅ Incidents saved to {csv_file}")
                
                # Also save to session state for immediate display in View Incident Logs
                if 'session_incidents' not in st.session_state:
                    st.session_state.session_incidents = []
                
                # Add new incidents from this session
                for inc in alert_system.incidents:
                    st.session_state.session_incidents.append({
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
                
            except Exception as e:
                st.error(f"❌ Error saving incidents: {str(e)}")

    st.session_state.stop_camera = False

def analyze_video(video_path, pose_detector, reba_scorer):
    """Analyze uploaded video file"""
    
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        st.error("Could not open video file")
        return None
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    results = []
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    frame_count = 0
    sample_rate = max(1, fps // 5)  # Sample 5 frames per second
    
    while cap.isOpened():
        ret, frame = cap.read()
        
        if not ret:
            break
        
        # Process only sampled frames to speed up analysis
        if frame_count % sample_rate == 0:
            result, error = process_image(frame, pose_detector, reba_scorer)
            
            if result:
                timestamp = frame_count / fps
                results.append({
                    'frame': frame_count,
                    'timestamp': timestamp,
                    'reba_score': result['reba_result']['score'],
                    'risk_level': result['reba_result']['risk_level'],
                    'trunk_angle': result['angles'].get('trunk_flexion'),
                    'neck_angle': result['angles'].get('neck_flexion')
                })
        
        frame_count += 1
        progress = min(frame_count / total_frames, 1.0)
        progress_bar.progress(progress)
        status_text.text(f"Analyzing frame {frame_count}/{total_frames}")
    
    cap.release()
    progress_bar.empty()
    status_text.empty()
    
    return results

def display_video_results(results):
    """Display video analysis results"""
    
    if not results:
        st.warning("No poses detected in video")
        return
    
    st.success(f"✅ Analyzed {len(results)} frames successfully!")
    
    # Convert to DataFrame
    df = pd.DataFrame(results)
    
    # Summary statistics
    st.subheader("📊 Video Analysis Summary")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        avg_reba = df['reba_score'].mean()
        st.metric("Average REBA Score", f"{avg_reba:.1f}")
    
    with col2:
        max_reba = df['reba_score'].max()
        st.metric("Maximum REBA Score", max_reba)
    
    with col3:
        high_risk_pct = (df['reba_score'] > 7).sum() / len(df) * 100
        st.metric("High Risk Frames", f"{high_risk_pct:.1f}%")
    
    # Risk distribution
    st.subheader("🎯 Risk Level Distribution")
    risk_counts = df['risk_level'].value_counts()
    
    import matplotlib.pyplot as plt
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Risk level pie chart
    colors = {'low': '#2ecc71', 'medium': '#f39c12', 'high': '#e67e22', 'very_high': '#e74c3c'}
    risk_colors = [colors.get(level, '#95a5a6') for level in risk_counts.index]
    
    ax1.pie(risk_counts.values, labels=risk_counts.index, autopct='%1.1f%%', 
            colors=risk_colors, startangle=90)
    ax1.set_title('Risk Level Distribution')
    
    # REBA score over time
    ax2.plot(df['timestamp'], df['reba_score'], linewidth=2)
    ax2.axhline(y=7, color='r', linestyle='--', label='High Risk Threshold')
    ax2.axhline(y=4, color='orange', linestyle='--', label='Medium Risk Threshold')
    ax2.set_xlabel('Time (seconds)')
    ax2.set_ylabel('REBA Score')
    ax2.set_title('REBA Score Over Time')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    st.pyplot(fig)
    
    # Detailed frame data
    with st.expander("📋 View Detailed Frame Data"):
        st.dataframe(df)
        
        # Download button
        csv = df.to_csv(index=False)
        st.download_button(
            label="⬇️ Download Results as CSV",
            data=csv,
            file_name="video_analysis_results.csv",
            mime="text/csv"
        )

def process_image(image, pose_detector, reba_scorer, load_weight=0, coupling_quality='good', static_image=False):
    """Process a single image and return results"""
    # Detect pose
    results = pose_detector.detect_pose(image, static_image=static_image)
    
    # Extract landmarks
    landmarks = pose_detector.extract_landmarks(results)
    
    if landmarks is None:
        # If debug info available from the detector, include a short summary
        debug_info = getattr(pose_detector, 'last_detection_debug', None)
        if debug_info:
            summaries = []
            for a in debug_info:
                r = a.get('results')
                count = 0
                max_vis = 0
                try:
                    if r and r.pose_landmarks:
                        count = len(r.pose_landmarks.landmark)
                        max_vis = max([getattr(l, 'visibility', 0.0) for l in r.pose_landmarks.landmark])
                except Exception:
                    pass
                summaries.append(f"{a.get('method')}: landmarks={count}, max_vis={max_vis:.2f}")
            summary_str = " | ".join(summaries)
            return None, f"No person detected in the image. Attempts: {summary_str}"
        return None, "No person detected in the image"
    
    # Calculate angles
    angles = extract_all_angles(landmarks)
    
    if angles is None:
        return None, "Unable to calculate joint angles"
    
    # Calculate REBA score with load and coupling
    reba_result = reba_scorer.calculate_reba_score(angles, load_weight=load_weight, coupling=coupling_quality)
    
    # Create annotated image
    annotated_image = image.copy()
    annotated_image = pose_detector.draw_landmarks(annotated_image, results)
    annotated_image = create_risk_visualization(
        annotated_image, 
        reba_result['score'], 
        reba_result['risk_level']
    )
    
    return {
        'reba_result': reba_result,
        'angles': angles,
        'landmarks': landmarks,
        'annotated_image': annotated_image
    }, None

def main():
    st.set_page_config(
        page_title="AI4HSSE - Ergonomic Posture Analysis",
        page_icon="⚕️",
        layout="wide"
    )
    
    st.title("🏗️ AI4HSSE: Ergonomic Posture Analysis System")
    st.markdown("**AI-Based Ergonomic Posture Analysis for Oil and Gas Field Workers**")
    
    # Load models
    pose_detector, reba_scorer = load_models()
    
    # Sidebar
    st.sidebar.title("Analysis Options")
    analysis_mode = st.sidebar.selectbox(
        "Select Analysis Mode",
        ["Upload Image", "Real-time Camera", "Upload Video", "View Incident Logs"]
    )
    
    # Alert settings in sidebar
    st.sidebar.subheader("⚙️ Alert Settings")
    alert_threshold = st.sidebar.slider("Alert Threshold (REBA)", 1, 15, 7, 
                                        help="Trigger alert when REBA score reaches this value (1-15)")
    alert_duration = st.sidebar.slider("Alert Duration (seconds)", 2.0, 10.0, 3.0, 0.5,
                                       help="How long to sustain high risk before alerting")
    # Camera display width (fixed pixel width to help see details)
    camera_width = st.sidebar.slider("Camera display width (px)", 320, 1280, 640, step=32,
                                     help="Width of the in-browser camera image (pixels)")
    # Debug toggle for alert diagnostics
    debug_alerts = st.sidebar.checkbox("Show alert debug", value=False, help="Display REBA and sustained duration during camera session")
    # Upload debug toggle: show landmark visibility and raw detection info for uploaded images
    upload_debug = st.sidebar.checkbox("Show upload debug info", value=False, help="When enabled, displays landmark presence/visibility and computed angles for uploaded images")
    # Very-high observations (REBA >= 11) will be logged automatically by the system
    
    # Load weight input
    load_weight = st.sidebar.slider("Load Weight (kg)", 0, 50, 0)
    coupling_quality = st.sidebar.selectbox("Coupling Quality", ["good", "fair", "poor"])
    
    if analysis_mode == "Upload Image":
        st.header("📸 Image Analysis")
        
        uploaded_file = st.file_uploader(
            "Choose an image...", 
            type=["jpg", "jpeg", "png"],
            help="Upload an image showing a person in a working posture"
        )
        
        if uploaded_file is not None:
            # Load and display image
            image = Image.open(uploaded_file)
            image_np = np.array(image)
            image_cv = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
            # Preprocess uploaded image to improve detection
            try:
                from utils import preprocess_image
                image_cv = preprocess_image(image_cv, max_width=1024, equalize=True)
            except Exception:
                pass
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Original Image")
                st.image(image, use_container_width=True)
            
            if st.button("🔍 Analyze Posture", type="primary"):
                with st.spinner("Analyzing posture..."):
                    results, error = process_image(image_cv, pose_detector, reba_scorer, load_weight, coupling_quality, static_image=True)
                
                if error:
                    st.error(f"Analysis failed: {error}")
                else:
                    # If upload debug enabled, show landmark visibility / raw info
                    if upload_debug:
                        try:
                            lm = results.get('landmarks')
                            if lm is None:
                                st.write("No landmarks detected")
                            else:
                                # Show summary of landmark visibility
                                vis_list = [float(v[3]) if len(v) > 3 else 0.0 for v in lm]
                                visible_count = sum(1 for v in vis_list if v > 0.5)
                                st.write(f"Landmarks detected: {len(lm)}; Visible (>0.5): {visible_count}")
                                st.write("Max visibility:", max(vis_list) if vis_list else 0)
                                # Draw landmark visibility overlay on the annotated image for visual debug
                                try:
                                    ann = results.get('annotated_image').copy()
                                    h, w = ann.shape[:2]
                                    for idx, lv in enumerate(lm):
                                        try:
                                            x = int(lv[0] * w)
                                            y = int(lv[1] * h)
                                            vis_val = float(lv[3]) if len(lv) > 3 else 0.0
                                            col = (0, 255, 0) if vis_val > 0.5 else (0, 0, 255)
                                            cv2.circle(ann, (x, y), 4, col, -1)
                                            cv2.putText(ann, f"{vis_val:.2f}", (x + 6, y + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.4, col, 1)
                                        except Exception:
                                            continue
                                    # Replace annotated image with debug overlay
                                    results['annotated_image'] = ann
                                except Exception as e:
                                    st.write("Could not draw debug overlay:", e)
                        except Exception as e:
                            st.write("Upload debug info unavailable:", e)

                    with col2:
                        st.subheader("Analysis Results")
                        
                        # Convert annotated image back to RGB for display
                        annotated_rgb = cv2.cvtColor(results['annotated_image'], cv2.COLOR_BGR2RGB)
                        st.image(annotated_rgb, use_container_width=True)
                    
                    # Display metrics
                    reba_result = results['reba_result']
                    risk_info = RISK_LEVELS[reba_result['risk_level']]
                    
                    # Metrics row
                    metric_col1, metric_col2, metric_col3 = st.columns(3)
                    
                    with metric_col1:
                        st.metric(
                            "REBA Score", 
                            reba_result['score'],
                            help="Rapid Entire Body Assessment score (1-15)"
                        )
                    
                    with metric_col2:
                        st.metric(
                            "Risk Level", 
                            reba_result['risk_level'].replace('_', ' ').title(),
                            help="Risk category based on REBA score"
                        )
                    
                    with metric_col3:
                        st.metric(
                            "Action Required",
                            "Yes" if reba_result['score'] > 3 else "Monitor",
                            help="Whether immediate action is needed"
                        )
                    
                    # Detailed results
                    st.subheader("📊 Detailed Analysis")
                    
                    detail_col1, detail_col2 = st.columns(2)
                    
                    with detail_col1:
                        st.write("**Joint Angles:**")
                        angles = results['angles']
                        if angles:
                            for angle_name, angle_value in angles.items():
                                if angle_value is not None:
                                    st.write(f"• {angle_name.replace('_', ' ').title()}: {angle_value:.1f}°")
                    
                    with detail_col2:
                        st.write("**REBA Component Scores:**")
                        details = reba_result['details']
                        st.write(f"• Trunk: {details.get('trunk_score', 'N/A')}")
                        st.write(f"• Neck: {details.get('neck_score', 'N/A')}")
                        st.write(f"• Upper Arm: {details.get('upper_arm_score', 'N/A')}")
                    
                    # Recommendations
                    st.subheader("💡 Recommendations")
                    st.info(f"**Action Required:** {risk_info['action']}")
                    
                    if reba_result['score'] > 7:
                        st.warning("⚠️ High risk posture detected! Consider immediate ergonomic interventions.")
                    elif reba_result['score'] > 3:
                        st.warning("⚠️ Medium risk posture. Monitor and consider improvements.")
                    else:
                        st.success("✅ Low risk posture. Continue monitoring.")
    
    elif analysis_mode == "Real-time Camera":
        st.header("📹 Real-time Camera Analysis")
        st.warning("⚠️ Make sure your full body is visible in the camera for accurate detection")
        
        if st.button("🎥 Start Real-Time Analysis", type="primary", key="start_cam"):
            with st.spinner("Starting camera..."):
                # Use Streamlit-friendly camera display (no native cv2 windows)
                camera_placeholder = st.empty()
                metrics_placeholder = st.empty()
                run_camera_analysis(
                    camera_placeholder,
                    metrics_placeholder,
                    pose_detector,
                    reba_scorer,
                    alert_threshold=alert_threshold,
                    alert_duration=alert_duration,
                    debug=debug_alerts,
                    camera_width=camera_width,
                    load_weight=load_weight,
                    coupling_quality=coupling_quality,
                )
        
    elif analysis_mode == "Upload Video":  # Video upload
        st.header("🎥 Video Analysis")
        
        uploaded_video = st.file_uploader(
            "Choose a video file...",
            type=["mp4", "avi", "mov", "mkv"],
            help="Upload a video showing work postures for analysis"
        )
        
        if uploaded_video is not None:
            # Save uploaded video temporarily
            temp_video_path = "temp_video.mp4"
            with open(temp_video_path, "wb") as f:
                f.write(uploaded_video.read())
            
            st.video(temp_video_path)
            
            if st.button("🔍 Analyze Video", type="primary"):
                with st.spinner("Analyzing video frames..."):
                    results = analyze_video(temp_video_path, pose_detector, reba_scorer)
                
                if results:
                    display_video_results(results)
                
                # Clean up
                if os.path.exists(temp_video_path):
                    os.remove(temp_video_path)
    
    elif analysis_mode == "View Incident Logs":  # View Incident Logs
        st.header("📋 Incident Logs & Reports")
        
        st.markdown("""
        View historical incident logs and generate compliance reports.
        All high-risk incidents are automatically logged during real-time monitoring.
        """)
        
        # Show current session incidents first
        if 'session_incidents' in st.session_state and len(st.session_state.session_incidents) > 0:
            st.subheader("📊 Current Session Incidents")
            
            session_df = pd.DataFrame(st.session_state.session_incidents)
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Incidents", len(session_df))
            
            with col2:
                total_duration = session_df['duration_seconds'].sum()
                st.metric("Total Exposure Time", f"{total_duration:.1f}s")
            
            with col3:
                avg_reba = session_df['peak_reba_score'].mean()
                st.metric("Avg REBA Score", f"{avg_reba:.1f}")
            
            with col4:
                max_reba = session_df['peak_reba_score'].max()
                st.metric("Max REBA Score", int(max_reba))
            
            st.dataframe(session_df, use_container_width=True)
            
            st.divider()
        
        # Check for log files
        log_files = []
        if os.path.exists('logs'):
            log_files = [f for f in os.listdir('logs') if f.startswith('incidents_') and f.endswith('.csv')]
        
        if not log_files and ('session_incidents' not in st.session_state or len(st.session_state.session_incidents) == 0):
            st.info("📝 No incident logs found yet. Start a camera session to begin logging.")
        elif log_files:
            # Date selector
            dates = [f.replace('incidents_', '').replace('.csv', '') for f in log_files]
            selected_date = st.selectbox("Select Date", sorted(dates, reverse=True))
            
            # Load and display log
            log_file = f'logs/incidents_{selected_date}.csv'
            df = pd.read_csv(log_file)
            
            # Summary metrics
            st.subheader(f"📊 Summary for {selected_date}")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Incidents", len(df))
            
            with col2:
                total_duration = df['duration_seconds'].sum()
                st.metric("Total Exposure Time", f"{total_duration:.1f}s")
            
            with col3:
                avg_reba = df['peak_reba_score'].mean()
                st.metric("Avg REBA Score", f"{avg_reba:.1f}")
            
            with col4:
                max_reba = df['peak_reba_score'].max()
                st.metric("Max REBA Score", max_reba)
            
            # Visualizations
            st.subheader("📈 Risk Analysis")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Risk level distribution
                import matplotlib.pyplot as plt
                
                fig, ax = plt.subplots(figsize=(8, 5))
                risk_counts = df['risk_level'].value_counts()
                
                colors = {'low': '#2ecc71', 'medium': '#f39c12', 'high': '#e67e22', 'very_high': '#e74c3c'}
                bar_colors = [colors.get(level, '#95a5a6') for level in risk_counts.index]
                
                ax.bar(risk_counts.index, risk_counts.values, color=bar_colors)
                ax.set_xlabel('Risk Level')
                ax.set_ylabel('Number of Incidents')
                ax.set_title('Incidents by Risk Level')
                ax.grid(True, alpha=0.3, axis='y')
                
                st.pyplot(fig)
            
            with col2:
                # Duration distribution
                fig, ax = plt.subplots(figsize=(8, 5))
                ax.hist(df['duration_seconds'], bins=15, edgecolor='black', alpha=0.7, color='#3498db')
                ax.set_xlabel('Duration (seconds)')
                ax.set_ylabel('Frequency')
                ax.set_title('Incident Duration Distribution')
                ax.grid(True, alpha=0.3, axis='y')
                
                st.pyplot(fig)
            
            # Detailed incident table
            st.subheader("📋 Detailed Incidents")
            
            # Format the dataframe for display
            display_df = df.copy()
            display_df['start_time'] = pd.to_datetime(display_df['start_time'], format='%H:%M:%S').dt.strftime('%H:%M:%S')
            display_df['end_time'] = pd.to_datetime(display_df['end_time'], format='%H:%M:%S').dt.strftime('%H:%M:%S')
            display_df['duration_seconds'] = display_df['duration_seconds'].round(1)
            
            st.dataframe(display_df, use_container_width=True)
            
            # Download button
            csv = df.to_csv(index=False)
            st.download_button(
                label="📥 Download CSV Report",
                data=csv,
                file_name=f"incident_report_{selected_date}.csv",
                mime="text/csv"
            )
            
            # HSE Compliance Report
            st.subheader("📄 HSE Compliance Summary")
            
            st.markdown(f"""
            **Report Date:** {selected_date}
            
            **Ergonomic Risk Assessment Summary:**
            - Total monitoring incidents: {len(df)}
            - High-risk postures detected: {len(df[df['risk_level'].isin(['high', 'very_high'])])}
            - Total exposure to high-risk postures: {df['duration_seconds'].sum():.1f} seconds
            - Average REBA score: {df['peak_reba_score'].mean():.1f}
            - Peak REBA score: {df['peak_reba_score'].max()}
            
            **Recommendations:**
            - {'✅ Risk levels are within acceptable limits' if df['peak_reba_score'].mean() < 6 else '⚠️ Ergonomic intervention recommended'}
            - {'Maintain current safety practices' if len(df) < 10 else 'Review work procedures to reduce high-risk postures'}
            - Continue monitoring and periodic assessment
            
            *This report was generated automatically by AI4HSSE System*
            """)
    
    # About section
    with st.expander("ℹ️ About AI4HSSE System"):
        st.write("""
        This AI-based ergonomic analysis system uses:
        - **MediaPipe BlazePose** for human pose detection
        - **REBA (Rapid Entire Body Assessment)** scoring methodology
        - **Computer vision** for automated posture analysis
        
        The system is designed specifically for oil and gas field workers to help:
        - Identify high-risk postures
        - Prevent musculoskeletal disorders (MSDs)
        - Improve workplace safety
        """)

if __name__ == "__main__":
    main()
