import streamlit as st
import cv2
import pandas as pd
import numpy as np
from PIL import Image, ImageDraw
import plotly.graph_objects as go
import plotly.express as px
from collections import defaultdict, deque
import json
import io
import base64
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Note: In a real implementation, you would use YOLO. For this demo, we'll simulate detection.
# from ultralytics import YOLO

# --- Configuration ---
st.set_page_config(
    page_title="Advanced Dodgeball AI Annotation",
    page_icon="🎯",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .annotation-header {
        background: linear-gradient(90deg, #667eea, #764ba2);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    .control-panel {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #667eea;
    }
    .stats-box {
        background: #e8f4fd;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
    }
    .event-log {
        max-height: 300px;
        overflow-y: auto;
        border: 1px solid #dee2e6;
        border-radius: 5px;
        padding: 10px;
    }
    .player-tag {
        display: inline-block;
        padding: 4px 8px;
        margin: 2px;
        border-radius: 15px;
        font-size: 12px;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

class MockYOLODetector:
    """Mock YOLO detector for demonstration purposes."""
    
    def __init__(self):
        self.track_id_counter = 0
        self.active_tracks = {}
        self.track_history = defaultdict(lambda: deque(maxlen=30))
    
    def detect_and_track(self, frame, frame_number):
        """Simulate person detection and tracking."""
        height, width = frame.shape[:2]
        
        # Simulate detection of 4-8 players
        num_players = np.random.randint(4, 9)
        detections = []
        
        for i in range(num_players):
            # Simulate realistic player positions
            x = np.random.randint(50, width - 100)
            y = np.random.randint(50, height - 100)
            w = np.random.randint(60, 120)
            h = np.random.randint(120, 200)
            
            # Ensure box is within frame
            x = max(0, min(x, width - w))
            y = max(0, min(y, height - h))
            
            confidence = np.random.uniform(0.6, 0.95)
            
            # Assign or maintain track ID
            center = (x + w//2, y + h//2)
            track_id = self._assign_track_id(center, frame_number)
            
            detection = {
                'box': [x, y, x + w, y + h],
                'confidence': confidence,
                'track_id': track_id,
                'center': center
            }
            detections.append(detection)
        
        return detections
    
    def _assign_track_id(self, center, frame_number):
        """Assign track ID based on proximity to previous detections."""
        min_distance = float('inf')
        assigned_id = None
        
        # Check existing tracks
        for track_id, history in self.track_history.items():
            if history:
                last_center = history[-1]['center']
                distance = np.sqrt((center[0] - last_center[0])**2 + (center[1] - last_center[1])**2)
                if distance < min_distance and distance < 100:  # Threshold for same person
                    min_distance = distance
                    assigned_id = track_id
        
        # Create new track if no match
        if assigned_id is None:
            assigned_id = f"T{self.track_id_counter}"
            self.track_id_counter += 1
        
        # Update track history
        self.track_history[assigned_id].append({
            'center': center,
            'frame': frame_number
        })
        
        return assigned_id

class AdvancedAnnotationTool:
    """Advanced annotation tool with comprehensive features."""
    
    def __init__(self):
        self.detector = MockYOLODetector()
        self.init_session_state()
    
    def init_session_state(self):
        """Initialize all session state variables."""
        defaults = {
            'events': [],
            'current_frame': 0,
            'video_cap': None,
            'video_name': None,
            'selected_action': "Throw",
            'nickname_map': {},
            'team_assignments': {},
            'player_stats': defaultdict(lambda: defaultdict(int)),
            'game_settings': {
                'game_name': f"Game_{datetime.now().strftime('%Y%m%d_%H%M')}",
                'team_names': ['Team A', 'Team B'],
                'match_duration': 300,  # seconds
                'current_set': 1
            },
            'auto_tracking': True,
            'detection_confidence': 0.5,
            'annotation_mode': 'Manual',
            'heatmap_data': defaultdict(list),
            'trajectory_data': defaultdict(list)
        }
        
        for key, value in defaults.items():
            if key not in st.session_state:
                st.session_state[key] = value
    
    def load_video(self, uploaded_file):
        """Load and initialize video."""
        if uploaded_file is not None:
            # Save uploaded file
            video_path = f"temp_{uploaded_file.name}"
            with open(video_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            # Initialize video capture
            cap = cv2.VideoCapture(video_path)
            if cap.isOpened():
                st.session_state.video_cap = cap
                st.session_state.video_name = uploaded_file.name
                
                # Reset states for new video
                st.session_state.current_frame = 0
                st.session_state.events = []
                st.session_state.nickname_map = {}
                st.session_state.team_assignments = {}
                st.session_state.player_stats = defaultdict(lambda: defaultdict(int))
                
                st.success(f"✅ Video '{uploaded_file.name}' loaded successfully!")
                return True
            else:
                st.error("Failed to load video. Please try a different file.")
                return False
        return False
    
    def process_frame(self):
        """Process current frame with AI detection."""
        if not st.session_state.video_cap:
            return None, []
        
        # Get current frame
        st.session_state.video_cap.set(cv2.CAP_PROP_POS_FRAMES, st.session_state.current_frame)
        ret, frame = st.session_state.video_cap.read()
        
        if not ret:
            return None, []
        
        # Detect and track objects
        detections = self.detector.detect_and_track(frame, st.session_state.current_frame)
        
        # Draw detections
        annotated_frame = frame.copy()
        for detection in detections:
            x1, y1, x2, y2 = detection['box']
            track_id = detection['track_id']
            confidence = detection['confidence']
            
            # Get display name
            nickname = st.session_state.nickname_map.get(track_id, track_id)
            team = st.session_state.team_assignments.get(track_id, 'Unassigned')
            
            # Choose color based on team
            if team == st.session_state.game_settings['team_names'][0]:
                color = (255, 0, 0)  # Red
            elif team == st.session_state.game_settings['team_names'][1]:
                color = (0, 0, 255)  # Blue
            else:
                color = (128, 128, 128)  # Gray
            
            # Draw bounding box
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 3)
            
            # Draw label
            label = f"{nickname} ({confidence:.2f})"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
            cv2.rectangle(annotated_frame, (x1, y1 - label_size[1] - 10), 
                         (x1 + label_size[0], y1), color, -1)
            cv2.putText(annotated_frame, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Store trajectory data
            center = detection['center']
            st.session_state.trajectory_data[track_id].append({
                'frame': st.session_state.current_frame,
                'x': center[0],
                'y': center[1],
                'timestamp': st.session_state.current_frame / 30  # Assuming 30 FPS
            })
        
        return annotated_frame, detections
    
    def log_event(self, player_id, action, additional_data=None):
        """Log an annotation event."""
        event = {
            'frame': st.session_state.current_frame,
            'timestamp': st.session_state.current_frame / 30,  # Assuming 30 FPS
            'player_id': player_id,
            'nickname': st.session_state.nickname_map.get(player_id, player_id),
            'team': st.session_state.team_assignments.get(player_id, 'Unassigned'),
            'action': action,
            'set': st.session_state.game_settings['current_set'],
            'game': st.session_state.game_settings['game_name']
        }
        
        if additional_data:
            event.update(additional_data)
        
        st.session_state.events.append(event)
        
        # Update player stats
        st.session_state.player_stats[player_id][action] += 1
        
        # Store heatmap data for position-based actions
        if player_id in st.session_state.trajectory_data:
            if st.session_state.trajectory_data[player_id]:
                last_pos = st.session_state.trajectory_data[player_id][-1]
                st.session_state.heatmap_data[action].append({
                    'x': last_pos['x'],
                    'y': last_pos['y'],
                    'player': player_id
                })
    
    def create_analytics_dashboard(self):
        """Create real-time analytics dashboard."""
        if not st.session_state.events:
            st.info("No events logged yet. Start annotating to see analytics!")
            return
        
        # Convert events to DataFrame
        events_df = pd.DataFrame(st.session_state.events)
        
        # Create dashboard
        col1, col2 = st.columns(2)
        
        with col1:
            # Action distribution
            action_counts = events_df['action'].value_counts()
            fig_actions = px.pie(values=action_counts.values, names=action_counts.index,
                               title="Action Distribution")
            st.plotly_chart(fig_actions, use_container_width=True)
            
            # Player performance
            player_stats = events_df.groupby(['nickname', 'action']).size().unstack(fill_value=0)
            if len(player_stats) > 0:
                fig_players = px.bar(player_stats, title="Player Action Counts",
                                   barmode='group')
                st.plotly_chart(fig_players, use_container_width=True)
        
        with col2:
            # Team comparison
            if 'team' in events_df.columns:
                team_stats = events_df.groupby(['team', 'action']).size().unstack(fill_value=0)
                if len(team_stats) > 0:
                    fig_teams = px.bar(team_stats, title="Team Performance",
                                     barmode='group')
                    st.plotly_chart(fig_teams, use_container_width=True)
            
            # Timeline of events
            events_df['minute'] = (events_df['timestamp'] / 60).astype(int)
            timeline = events_df.groupby(['minute', 'action']).size().unstack(fill_value=0)
            if len(timeline) > 0:
                fig_timeline = px.area(timeline, title="Events Timeline (by minute)")
                st.plotly_chart(fig_timeline, use_container_width=True)
    
    def export_annotations(self):
        """Export annotations in multiple formats."""
        if not st.session_state.events:
            st.warning("No events to export!")
            return
        
        # Prepare data
        events_df = pd.DataFrame(st.session_state.events)
        
        # Summary statistics
        summary_stats = {}
        for player_id, stats in st.session_state.player_stats.items():
            nickname = st.session_state.nickname_map.get(player_id, player_id)
            team = st.session_state.team_assignments.get(player_id, 'Unassigned')
            summary_stats[nickname] = {
                'Team': team,
                'Throws': stats.get('Throw', 0),
                'Hits': stats.get('Hit', 0),
                'Catches': stats.get('Catch', 0),
                'Dodges': stats.get('Dodge', 0),
                'Blocks': stats.get('Block', 0)
            }
        
        summary_df = pd.DataFrame(summary_stats).T
        
        # Export options
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # CSV export
            csv_events = events_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Events CSV",
                data=csv_events,
                file_name=f"events_{st.session_state.game_settings['game_name']}.csv",
                mime="text/csv"
            )
        
        with col2:
            # Summary CSV
            csv_summary = summary_df.to_csv()
            st.download_button(
                label="📊 Download Summary CSV",
                data=csv_summary,
                file_name=f"summary_{st.session_state.game_settings['game_name']}.csv",
                mime="text/csv"
            )
        
        with col3:
            # JSON export with full data
            export_data = {
                'game_info': st.session_state.game_settings,
                'events': st.session_state.events,
                'player_mappings': {
                    'nicknames': st.session_state.nickname_map,
                    'teams': st.session_state.team_assignments
                },
                'statistics': dict(st.session_state.player_stats)
            }
            
            json_data = json.dumps(export_data, indent=2)
            st.download_button(
                label="💾 Download Full JSON",
                data=json_data,
                file_name=f"full_data_{st.session_state.game_settings['game_name']}.json",
                mime="application/json"
            )

def main():
    """Main application."""
    # Header
    st.markdown("""
    <div class="annotation-header">
        <h1>🎯 Advanced AI Dodgeball Annotation Tool</h1>
        <p>Professional-grade video analysis with real-time player tracking and comprehensive analytics</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize tool
    tool = AdvancedAnnotationTool()
    
    # Sidebar configuration
    st.sidebar.header("🎛️ Configuration")
    
    # Game settings
    with st.sidebar.expander("🏆 Game Settings", expanded=True):
        st.session_state.game_settings['game_name'] = st.text_input(
            "Game Name", value=st.session_state.game_settings['game_name'])
        
        st.session_state.game_settings['team_names'][0] = st.text_input(
            "Team A Name", value=st.session_state.game_settings['team_names'][0])
        
        st.session_state.game_settings['team_names'][1] = st.text_input(
            "Team B Name", value=st.session_state.game_settings['team_names'][1])
        
        st.session_state.game_settings['current_set'] = st.number_input(
            "Current Set", min_value=1, max_value=10, 
            value=st.session_state.game_settings['current_set'])
    
    # AI settings
    with st.sidebar.expander("🤖 AI Settings"):
        st.session_state.auto_tracking = st.checkbox("Auto-tracking", value=True)
        st.session_state.detection_confidence = st.slider(
            "Detection Confidence", 0.1, 1.0, 0.5)
        st.session_state.annotation_mode = st.selectbox(
            "Annotation Mode", ["Manual", "Semi-Automatic", "Automatic"])
    
    # Main interface
    tab1, tab2, tab3, tab4 = st.tabs(["📹 Video Annotation", "👥 Player Management", 
                                     "📊 Live Analytics", "💾 Export Data"])
    
    with tab1:
        st.header("Video Annotation Interface")
        
        # Video upload
        uploaded_file = st.file_uploader(
            "Upload Dodgeball Game Video", 
            type=["mp4", "mov", "avi", "mkv"],
            help="Supported formats: MP4, MOV, AVI, MKV"
        )
        
        if tool.load_video(uploaded_file) or st.session_state.video_cap:
            # Video controls
            total_frames = int(st.session_state.video_cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = st.session_state.video_cap.get(cv2.CAP_PROP_FPS)
            duration = total_frames / fps if fps > 0 else 0
            
            col1, col2 = st.columns([3, 1])
            
            with col1:
                # Process and display frame
                frame, detections = tool.process_frame()
                
                if frame is not None:
                    # Convert BGR to RGB for display
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    st.image(frame_rgb, caption=f"Frame {st.session_state.current_frame}")
                    
                    # Click detection simulation
                    if detections and st.button("🎯 Click to Annotate Detected Player"):
                        # Simulate clicking on first detected player
                        selected_player = detections[0]['track_id']
                        tool.log_event(selected_player, st.session_state.selected_action)
                        st.success(f"Logged '{st.session_state.selected_action}' for {selected_player}")
            
            with col2:
                st.markdown('<div class="control-panel">', unsafe_allow_html=True)
                st.subheader("⚡ Controls")
                
                # Frame navigation
                st.session_state.current_frame = st.slider(
                    "Frame", 0, max(0, total_frames - 1), st.session_state.current_frame)
                
                # Navigation buttons
                nav_col1, nav_col2 = st.columns(2)
                with nav_col1:
                    if st.button("⏪ -10"):
                        st.session_state.current_frame = max(0, st.session_state.current_frame - 10)
                        st.rerun()
                    if st.button("⬅️ -1"):
                        st.session_state.current_frame = max(0, st.session_state.current_frame - 1)
                        st.rerun()
                
                with nav_col2:
                    if st.button("➡️ +1"):
                        st.session_state.current_frame = min(total_frames - 1, st.session_state.current_frame + 1)
                        st.rerun()
                    if st.button("⏩ +10"):
                        st.session_state.current_frame = min(total_frames - 1, st.session_state.current_frame + 10)
                        st.rerun()
                
                # Video info
                st.info(f"""
                **Video Info:**
                - Frame: {st.session_state.current_frame}/{total_frames-1}
                - Time: {st.session_state.current_frame/fps:.1f}s/{duration:.1f}s
                - FPS: {fps:.1f}
                """)
                
                # Action selection
                st.subheader("🎮 Action Selection")
                actions = ["Throw", "Hit", "Catch", "Dodge", "Block", "Elimination", "Revival"]
                st.session_state.selected_action = st.selectbox("Action to Log:", actions)
                
                # Quick stats
                if st.session_state.events:
                    st.subheader("📈 Quick Stats")
                    total_events = len(st.session_state.events)
                    unique_players = len(set(e['player_id'] for e in st.session_state.events))
                    st.metric("Total Events", total_events)
                    st.metric("Active Players", unique_players)
                
                st.markdown('</div>', unsafe_allow_html=True)
            
            # Recent events log
            if st.session_state.events:
                st.subheader("📝 Recent Events")
                recent_events = st.session_state.events[-10:][::-1]  # Last 10, reversed
                
                for event in recent_events:
                    st.markdown(f"""
                    <div class="stats-box">
                        <strong>{event['nickname']}</strong> ({event['team']}) 
                        - <em>{event['action']}</em> 
                        @ {event['timestamp']:.1f}s
                    </div>
                    """, unsafe_allow_html=True)
    
    with tab2:
        st.header("Player Management")
        
        # Display detected players
        if st.session_state.video_cap:
            frame, detections = tool.process_frame()
            
            if detections:
                st.subheader("🔍 Detected Players")
                
                for i, detection in enumerate(detections):
                    track_id = detection['track_id']
                    
                    with st.expander(f"Player {track_id} (Confidence: {detection['confidence']:.2f})"):
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            # Nickname assignment
                            nickname = st.text_input(
                                "Nickname:", 
                                value=st.session_state.nickname_map.get(track_id, ""),
                                key=f"nick_{track_id}"
                            )
                            
                            if st.button("Assign Nickname", key=f"assign_{track_id}"):
                                if nickname:
                                    st.session_state.nickname_map[track_id] = nickname
                                    st.success(f"Assigned nickname '{nickname}'")
                                else:
                                    if track_id in st.session_state.nickname_map:
                                        del st.session_state.nickname_map[track_id]
                        
                        with col2:
                            # Team assignment
                            team = st.selectbox(
                                "Team:",
                                ['Unassigned'] + st.session_state.game_settings['team_names'],
                                index=0,
                                key=f"team_{track_id}"
                            )
                            
                            if st.button("Assign Team", key=f"team_assign_{track_id}"):
                                st.session_state.team_assignments[track_id] = team
                                st.success(f"Assigned to {team}")
                
                # Bulk operations
                st.subheader("🔄 Bulk Operations")
                if st.button("Clear All Nicknames"):
                    st.session_state.nickname_map = {}
                    st.success("All nicknames cleared")
                
                if st.button("Clear All Team Assignments"):
                    st.session_state.team_assignments = {}
                    st.success("All team assignments cleared")
        else:
            st.info("Upload a video to start player management")
    
    with tab3:
        st.header("Live Analytics Dashboard")
        tool.create_analytics_dashboard()
        
        # Additional analytics
        if st.session_state.events:
            st.subheader("🔥 Heatmaps & Trajectories")
            
            # Action heatmap
            if st.session_state.heatmap_data:
                action_type = st.selectbox("Select Action for Heatmap:", 
                                         list(st.session_state.heatmap_data.keys()))
                
                if action_type in st.session_state.heatmap_data:
                    heatmap_points = st.session_state.heatmap_data[action_type]
                    if heatmap_points:
                        df_heatmap = pd.DataFrame(heatmap_points)
                        fig_heatmap = px.density_heatmap(
                            df_heatmap, x='x', y='y', 
                            title=f"{action_type} Heatmap",
                            color_continuous_scale='viridis'
                        )
                        st.plotly_chart(fig_heatmap, use_container_width=True)
    
    with tab4:
        st.header("Export Data")
        tool.export_annotations()
        
        # Import functionality
        st.subheader("📤 Import Previous Session")
        imported_file = st.file_uploader("Import JSON Data", type=['json'])
        
        if imported_file and st.button("Import Data"):
            try:
                data = json.load(imported_file)
                
                # Load data into session state
                st.session_state.events = data.get('events', [])
                st.session_state.nickname_map = data.get('player_mappings', {}).get('nicknames', {})
                st.session_state.team_assignments = data.get('player_mappings', {}).get('teams', {})
                st.session_state.game_settings.update(data.get('game_info', {}))
                
                # Rebuild stats
                st.session_state.player_stats = defaultdict(lambda: defaultdict(int))
                for event in st.session_state.events:
                    st.session_state.player_stats[event['player_id']][event['action']] += 1
                
                st.success("Data imported successfully!")
                st.rerun()
                
            except Exception as e:
                st.error(f"Import failed: {str(e)}")

if __name__ == "__main__":
    main()