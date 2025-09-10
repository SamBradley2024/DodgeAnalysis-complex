import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
import numpy as np
import warnings
import gspread
from google.oauth2.service_account import Credentials

warnings.filterwarnings('ignore')

# --- Styling and UI Helpers (No changes here) ---
def load_css():
    """Returns the custom CSS string."""
    return """
    <style>
        .main-header {
            background: linear-gradient(90deg, #FF6B6B, #4ECDC4); padding: 2rem; border-radius: 10px;
            margin-bottom: 2rem; color: white; text-align: center;
        }
        .metric-container {
            background: #f8f9fa; padding: 1rem; border-radius: 8px;
            border-left: 4px solid #4ECDC4; margin: 0.5rem 0;
        }
        .insight-box {
            background: #e8f4fd; padding: 1rem; border-radius: 8px;
            border-left: 4px solid #1f77b4; margin: 1rem 0;
            color: #31333F;
        }
        .warning-box {
            background: #fff3cd; padding: 1rem; border-radius: 8px;
            border-left: 4px solid #ffc107; margin: 1rem 0;
        }
    </style>
    """

def styled_metric(label, value, help_text=""):
    """Creates a styled metric box using custom CSS."""
    st.markdown(f'<div class="metric-container" title="{help_text}">', unsafe_allow_html=True)
    st.metric(label, value)
    st.markdown('</div>', unsafe_allow_html=True)

# --- Google Sheets Integration (No changes here) ---
@st.cache_resource(ttl=600)
def get_gspread_client():
    scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/spreadsheets", "https://www.googleapis.com/auth/drive.file", "https://www.googleapis.com/auth/drive"]
    creds_dict = st.secrets["gcp_service_account"]
    creds = Credentials.from_service_account_info(creds_dict, scopes=scope)
    client = gspread.authorize(creds)
    return client

def get_worksheet_names():
    try:
        client = get_gspread_client()
        spreadsheet = client.open("Dodgeball App Data")
        return [sheet.title for sheet in spreadsheet.worksheets()]
    except Exception as e:
        st.error(f"Could not connect to Google Sheets. Error: {e}")
        return []

# --- NEW: Multi-Game Data Loading and Processing Functions ---

def _process_single_dataframe(df, game_id, team_name_source):
    """A helper function to process a raw dataframe from a single source."""
    rename_map = {
        'Sets Played': 'Sets_Played', 'Hits (singles)': 'Hits_Singles', 'Throws (singles)': 'Throws_Singles',
        'Hits (multi-ball)': 'Hits_Multi', 'Throws (multi-ball)': 'Throws_Multi', 'Hits (counters)': 'Hits_Counters',
        'Throws (counters)': 'Throws_Counters', 'Hits (pres)': 'Hits_Pres', 'Throws (pres)': 'Throws_Pres',
        'Overall Hits': 'Hits', 'Overall Throws': 'Throws', 'Catches made': 'Catches', 'Catches attempted': 'Catches_Attempted',
        'Out (single hit)': 'Out_Single_Hit', 'Out (multi hit)': 'Out_Multi_Hit', 'Out (counter hit)': 'Out_Counter_Hit',
        'Out (pre hit)': 'Out_Pre_Hit', 'Out (overall hits)': 'Hit_Out', 'Out (caught)': 'Caught_Out',
        'Out (other)': 'Out_Other', 'Overall Outs': 'Times_Eliminated', 'Dodges (singles)': 'Dodges_Singles',
        'Dodges (multi-ball)': 'Dodges_Multi', 'Dodges (counters)': 'Dodges_Counters', 'Dodges (pres)': 'Dodges_Pres',
        'Dodges (overall)': 'Dodges', 'Blocks (singles)': 'Blocks_Singles', 'Blocks (multi-ball)': 'Blocks_Multi',
        'Blocks (counters)': 'Blocks_Counters', 'Blocks (pres)': 'Blocks_Pres', 'Blocks (overall)': 'Blocks',
        'Times Thrown At (singles)': 'Thrown_At_Singles', 'Times Thrown At (multi-ball)': 'Thrown_At_Multi',
        'Times Thrown At (counters)': 'Thrown_At_Counters', 'Times Thrown At (pres)': 'Thrown_At_Pres',
        'Times Thrown At (overall)': 'Thrown_At_Overall'
    }
    df = df.rename(columns=rename_map)
    
    team_name = team_name_source.split(' vs ')[0]
    df['Team'] = team_name
    df['Match_ID'] = 'M1' # Assume all loaded games are from the same match for now
    df['Game_ID'] = game_id # Use the sheet/file name as the unique Game ID
    df['Game_Outcome'] = 'Win' # Default value
    
    return df

def load_and_process_multiple_sheets(sheet_names):
    """Loads and processes multiple Google Sheet worksheets, combining them into one DataFrame."""
    all_game_dfs = []
    client = get_gspread_client()
    spreadsheet = client.open("Dodgeball App Data")
    
    for name in sheet_names:
        try:
            worksheet = spreadsheet.worksheet(name)
            data = worksheet.get_all_values()
            if not data or len(data) < 2: continue

            header_row = data[0]
            player_names = header_row[1:]
            metric_rows = data[1:]
            
            processed_data = {'Player_ID': player_names}
            for row in metric_rows:
                metric_name = row[0]
                metric_values = row[1:]
                if len(metric_values) == len(player_names):
                    processed_data[metric_name] = metric_values
            
            game_df = pd.DataFrame(processed_data)
            processed_df = _process_single_dataframe(game_df, name, header_row[0])
            all_game_dfs.append(processed_df)
        except Exception as e:
            st.warning(f"Could not process sheet '{name}'. Error: {e}")

    return pd.concat(all_game_dfs, ignore_index=True) if all_game_dfs else None

def load_and_process_multiple_csvs(uploaded_files):
    """Loads and processes multiple CSV files, combining them into one DataFrame."""
    all_game_dfs = []
    for file in uploaded_files:
        try:
            df_raw = pd.read_csv(file, header=None)
            if df_raw.empty or len(df_raw) < 2: continue

            header_row = df_raw.iloc[0].tolist()
            player_names = header_row[1:]
            metric_rows = df_raw.iloc[1:].values.tolist()

            processed_data = {'Player_ID': player_names}
            for row in metric_rows:
                metric_name = row[0]
                metric_values = row[1:]
                if len(metric_values) == len(player_names):
                    processed_data[metric_name] = metric_values
            
            game_df = pd.DataFrame(processed_data)
            processed_df = _process_single_dataframe(game_df, file.name, header_row[0])
            all_game_dfs.append(processed_df)
        except Exception as e:
            st.warning(f"Could not process file '{file.name}'. Error: {e}")
            
    return pd.concat(all_game_dfs, ignore_index=True) if all_game_dfs else None

# --- Data Enhancement & Model Training (No changes here, but now operates on multi-game data) ---
def enhance_dataframe(df):
    """Takes a raw dataframe and adds all calculated metrics and features."""
    required_cols = ['Player_ID', 'Team', 'Game_ID', 'Hits', 'Throws', 'Times_Eliminated']
    if not all(col in df.columns for col in required_cols):
        missing = [col for col in required_cols if col not in df.columns]
        st.error(f"Data enhancement failed. The data is missing required columns: {', '.join(missing)}")
        return None

    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    df['K/D_Ratio'] = df['Hits'] / df['Times_Eliminated'].replace(0, 1)
    df['Net_Impact'] = (df.get('Hits', 0) + df.get('Catches', 0)) - df.get('Times_Eliminated', 0)
    df['Hit_Accuracy'] = df['Hits'] / df['Throws'].replace(0, 1)
    df['Defensive_Efficiency'] = (df.get('Catches', 0) + df.get('Dodges', 0)) / (df.get('Catches', 0) + df.get('Dodges', 0) + df.get('Hit_Out', 0)).replace(0, 1)
    df['Offensive_Rating'] = (df.get('Hits', 0) * 2 + df.get('Throws', 0) * 0.5) / (df.get('Throws', 0) + 1)
    df['Defensive_Rating'] = (df.get('Dodges', 0) + df.get('Catches', 0) * 2) / 3
    df['Overall_Performance'] = (df['Offensive_Rating'] * 0.35 + df['Defensive_Rating'] * 0.35 + df['K/D_Ratio'] * 0.15 + df['Net_Impact'] * 0.05 + df['Hit_Accuracy'] * 0.05 + df['Defensive_Efficiency'] * 0.05)
    
    return df

def train_advanced_models(df):
    """Trains ML models and generates player roles."""
    df_copy = df.copy()
    models = {}
    
    role_features = ['Hits', 'Throws', 'Dodges', 'Catches', 'Hit_Accuracy', 'Defensive_Efficiency', 'Offensive_Rating', 'Defensive_Rating', 'K/D_Ratio']
    existing_role_features = [f for f in role_features if f in df_copy.columns]
    
    player_avg_stats = df_copy.groupby('Player_ID')[existing_role_features].mean()

    if player_avg_stats.empty or len(player_avg_stats) < 4:
        df_copy['Player_Role'] = 'N/A'
        return df_copy, models

    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(player_avg_stats)
    kmeans = KMeans(n_clusters=4, random_state=42, n_init='auto')
    
    player_avg_stats['Role_Cluster'] = kmeans.fit_predict(scaled_features)
    
    # Map roles back to the original dataframe
    df_copy = df_copy.join(player_avg_stats['Role_Cluster'], on='Player_ID')

    # ... (rest of the role naming logic can be complex, simplifying for now)
    role_mapping = {0: 'Offensive Heavy', 1: 'Defensive Wall', 2: 'All-Rounder', 3: 'Evasive Specialist'}
    df_copy['Player_Role'] = df_copy['Role_Cluster'].map(role_mapping).fillna('Generalist')
    
    return df_copy, models


def initialize_app(df, source_name):
    """Initializes the app by processing data and training models."""
    df_enhanced = enhance_dataframe(df.copy())
    if df_enhanced is not None:
        df_trained, models = train_advanced_models(df_enhanced)
        st.session_state.df_enhanced = df_trained
        st.session_state.models = models
        st.session_state.data_loaded = True
        st.session_state.source_name = source_name
        return True
    return False

# --- Visualization and Coaching Functions ---
def generate_advanced_coaching_report(df, player_id):
    """
    Generates a highly detailed coaching report with two charts: one for league
    percentiles and one comparing vs. the player's specific role average.
    """
    player_games = df[df['Player_ID'] == player_id]
    if player_games.empty: return [], [], None, None
    player_stats = player_games.mean(numeric_only=True)
    player_stats.name = player_games.index[0] # for .loc access later

    stats_to_analyze = [
        'Overall_Performance', 'K/D_Ratio', 'Hits', 'Throws', 'Catches', 'Dodges', 
        'Hits_Singles', 'Throws_Singles', 'Hits_Multi', 'Throws_Multi'
    ]
    existing_stats = [s for s in stats_to_analyze if s in df.columns and pd.api.types.is_numeric_dtype(df[s])]

    # --- 1. Percentile Analysis vs. League (based on player averages) ---
    player_averages = df.groupby('Player_ID')[existing_stats].mean()
    player_percentiles = player_averages.rank(pct=True).loc[player_id]
    
    percentile_diff = (player_percentiles - 0.5)
    top_strengths = percentile_diff.nlargest(3)
    top_weaknesses = percentile_diff.nsmallest(3)
    
    strengths_report, weaknesses_report = [], []
    for stat, value in top_strengths.items():
        if value > 0:
            strengths_report.append(f"**{stat.replace('_', ' ').title()}:** Ranks in the top **{100 - (player_percentiles[stat] * 100):.0f}%** of the league.")
    for stat, value in top_weaknesses.items():
        if value < 0:
            weaknesses_report.append(f"**{stat.replace('_', ' ').title()}:** Ranks in the bottom **{player_percentiles[stat] * 100:.0f}%** of the league.")

    if not strengths_report: strengths_report.append("No significant strengths identified.")
    if not weaknesses_report: weaknesses_report.append("No significant weaknesses identified.")

    # --- 2. Chart vs. League Average (Percentiles) ---
    fig_league = go.Figure()
    fig_league.add_trace(go.Scatterpolar(r=player_percentiles.values * 100, theta=player_percentiles.index.str.replace('_', ' '), fill='toself', name=f'{player_id} Percentile'))
    fig_league.add_trace(go.Scatterpolar(r=[50] * len(player_percentiles), theta=player_percentiles.index.str.replace('_', ' '), mode='lines', name='League Average (50th)', line=dict(dash='dash')))
    fig_league.update_layout(height=500, polar=dict(radialaxis=dict(visible=True, range=[0, 100])))

    # --- 3. Comparison vs. Role Average (Raw Stats) ---
    fig_role = None
    player_role = player_games['Player_Role'].iloc[0] if 'Player_Role' in player_games.columns else 'N/A'
    
    if player_role != 'N/A' and 'Player_Role' in df.columns:
        role_df = df[df['Player_Role'] == player_role]
        if len(role_df) > 1:
            role_avg_stats = role_df[existing_stats].mean()
            player_raw_avg_stats = player_averages.loc[player_id]
            
            fig_role = go.Figure()
            fig_role.add_trace(go.Bar(name=f'{player_id} (Avg)', x=existing_stats, y=player_raw_avg_stats))
            fig_role.add_trace(go.Bar(name=f'{player_role} Avg.', x=existing_stats, y=role_avg_stats))
            fig_role.update_layout(barmode='group', title_text=f'<b>Player Average Stats vs. Average "{player_role}"</b>', height=500)

    return strengths_report, weaknesses_report, fig_league, fig_role



def create_game_level_features(df):
    """Transforms player-level data into game-level data for win prediction."""
    team_game_stats = df.groupby(['Game_ID', 'Team']).agg(
        Avg_Performance=('Overall_Performance', 'mean'),
        Avg_KD_Ratio=('K/D_Ratio', 'mean'),
        Avg_Hit_Accuracy=('Hit_Accuracy', 'mean'),
        Win=('Game_Outcome', lambda x: 1 if (x == 'Win').any() else 0)
    ).reset_index()

    game_teams = team_game_stats.groupby('Game_ID')['Team'].apply(list).reset_index()
    game_teams = game_teams[game_teams['Team'].apply(len) == 2]

    game_features = []
    for _, row in game_teams.iterrows():
        game_id, (team_a_name, team_b_name) = row['Game_ID'], row['Team']
        
        team_a_stats = team_game_stats[(team_game_stats['Game_ID'] == game_id) & (team_game_stats['Team'] == team_a_name)].iloc[0]
        team_b_stats = team_game_stats[(team_game_stats['Game_ID'] == game_id) & (team_game_stats['Team'] == team_b_name)].iloc[0]
        
        feature_row = {
            'Team_A_Avg_Perf': team_a_stats['Avg_Performance'], 'Team_A_Avg_KD': team_a_stats['Avg_KD_Ratio'],
            'Team_A_Avg_Acc': team_a_stats['Avg_Hit_Accuracy'], 'Team_B_Avg_Perf': team_b_stats['Avg_Performance'],
            'Team_B_Avg_KD': team_b_stats['Avg_KD_Ratio'], 'Team_B_Avg_Acc': team_b_stats['Avg_Hit_Accuracy'],
            'Team_A_Won': team_a_stats['Win']
        }
        game_features.append(feature_row)
        
    return pd.DataFrame(game_features)

def train_win_prediction_model(game_level_df):
    """Trains a model to predict game outcomes."""
    if game_level_df.shape[0] < 10:
        st.warning("Not enough game data to train a win prediction model.")
        return None, None
        
    features = [col for col in game_level_df.columns if col != 'Team_A_Won']
    X, y = game_level_df[features], game_level_df['Team_A_Won']

    if len(y.unique()) < 2:
        st.warning("Not enough outcome variation to train a win prediction model.")
        return None, None

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
    
    model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    model.fit(X_train, y_train)
    
    st.session_state.win_model_accuracy = model.score(X_test, y_test)
    
    return model, features

def create_improvement_chart(df, player_id, metric_to_plot='Overall_Performance'): # <-- MODIFIED: Added new argument
    """Creates a line chart showing a player's performance over time with a trendline."""
    player_df = df[df['Player_ID'] == player_id].copy()
    
    # Calculate average of the chosen metric per match
    match_performance = player_df.groupby('Match_ID')[metric_to_plot].mean().reset_index() # <-- MODIFIED: Uses the new argument
    
    # Ensure matches are sorted for a proper time-series plot
    match_performance = match_performance.sort_values('Match_ID')
    
    # Guard Clause: Check if there's enough data for a trendline
    if len(match_performance) < 2:
        return None 

    # Create a numeric column for the x-axis to allow trendline calculation.
    match_performance['match_num'] = range(len(match_performance))
    
    # Create a nicely formatted name for titles and labels
    metric_name_formatted = metric_to_plot.replace('_', ' ').title()
    
    # Use the new numeric column for the 'x' axis and the chosen metric for 'y'.
    fig = px.scatter(
        match_performance,
        x='match_num',
        y=metric_to_plot, # <-- MODIFIED: Uses the new argument
        title=f'{player_id} - {metric_name_formatted} Trend Across Matches', # <-- MODIFIED: Dynamic title
        labels={"match_num": "Match", metric_to_plot: f"Average {metric_name_formatted}"}, # <-- MODIFIED: Dynamic labels
        trendline="ols",
        trendline_color_override="red"
    )
    
    # Update the x-axis tick labels to show the original string 'Match_ID's.
    fig.update_layout(
        xaxis = dict(
            tickmode = 'array',
            tickvals = match_performance['match_num'],
            ticktext = match_performance['Match_ID']
        )
    )
    
    # Update the main trace to show lines connecting the markers
    fig.update_traces(mode='lines+markers')
    
    return fig

def create_stamina_chart(df, player_id):
    """Creates a chart to visualize a player's performance across games in matches."""
    player_match_data = df[df['Player_ID'] == player_id].copy()
    if player_match_data.empty or player_match_data['Match_ID'].nunique() < 2:
        return None # Not enough data to plot stamina

    fig = px.line(
        player_match_data,
        x='Game_Num_In_Match',
        y='Overall_Performance',
        color='Match_ID',
        markers=True,
        title=f'{player_id} - Performance Trend Within Matches (Stamina)',
        labels={
            "Game_Num_In_Match": "Game Number in Match",
            "Overall_Performance": "Game Performance Score"
        }
    )
    fig.update_layout(xaxis_title="Game Number in Match", yaxis_title="Performance Score")
    return fig

def create_player_dashboard(df, player_id):
    """Create comprehensive player dashboard with multiple visualizations."""
    player_data = df[df['Player_ID'] == player_id]
    if player_data.empty:
        st.error(f"No data found for player {player_id}")
        return None

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Performance Radar', 'Game-by-Game Performance', 'Skill Distribution', 'Win Rate Analysis'),
        specs=[[{"type": "polar"}, {"type": "scatter"}], [{"type": "bar"}, {"type": "pie"}]]
    )
    
    radar_stats = ['Hits', 'Throws', 'Catches', 'Dodges', 'Blocks', 'K/D_Ratio']
    avg_radar_stats = player_data[radar_stats].mean()
    fig.add_trace(go.Scatterpolar(
        r=avg_radar_stats.values, theta=radar_stats,
        fill='toself', name='Avg Skills', line_color='#FF6B6B'
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=player_data['Game_ID'], y=player_data['Overall_Performance'],
        mode='lines+markers', name='Performance Trend', line=dict(color='#4ECDC4', width=3)
    ), row=1, col=2)

    bar_stats_cols = ['Hits', 'Throws', 'Catches', 'Dodges', 'Blocks']
    avg_bar_stats = player_data[bar_stats_cols].mean()
    fig.add_trace(go.Bar(
        x=bar_stats_cols, y=avg_bar_stats.values,
        name='Average Stats', marker_color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FECA57']
    ), row=2, col=1)

    outcomes = player_data['Game_Outcome'].value_counts()
    fig.add_trace(go.Pie(
        labels=outcomes.index, values=outcomes.values,
        name="Win Rate", marker_colors=['#4ECDC4', '#FF6B6B']
    ), row=2, col=2)
    fig.update_layout(height=800, showlegend=False, title_text=f"Comprehensive Dashboard: {player_id}")

    return fig

def create_team_analytics(df, team_id):
    """Create detailed team analytics visualization."""
    team_data = df[df['Team'] == team_id]
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Team Performance Distribution', 'Player Roles', 'Game Outcomes', 'Offensive vs. Defensive Rating'),
        specs=[[{"type": "histogram"}, {"type": "bar"}], [{"type": "pie"}, {"type": "scatter"}]]
    )
    fig.add_trace(go.Histogram(x=team_data['Overall_Performance'], nbinsx=15, name='Performance Distribution', marker_color='#4ECDC4'), row=1, col=1)
    if 'Player_Role' in team_data.columns:
        role_counts = team_data['Player_Role'].dropna().value_counts()
        if not role_counts.empty:
            fig.add_trace(go.Bar(x=role_counts.index, y=role_counts.values, name='Player Roles', marker_color='#FF6B6B'), row=1, col=2)
    outcomes = team_data['Game_Outcome'].value_counts()
    fig.add_trace(go.Pie(labels=outcomes.index, values=outcomes.values, name="Outcomes"), row=2, col=1)
    fig.add_trace(go.Scatter(x=team_data['Offensive_Rating'], y=team_data['Defensive_Rating'], mode='markers', text=team_data['Player_ID'], name='Off vs Def Rating', marker=dict(size=10, color=team_data['Overall_Performance'], colorscale='Viridis', showscale=True)), row=2, col=2)
    fig.update_layout(height=800, title_text=f"Team Analytics: {team_id}", showlegend=False)
    return fig

def create_league_overview(df):
    """Create comprehensive league overview."""
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Top Performers by Avg Score', 'Team Skill Comparison', 'League Role Distribution', 'Performance vs Consistency'),
        specs=[[{"type": "bar"}, {"type": "polar"}], [{"type": "pie"}, {"type": "scatter"}]]
    )
    
    top_players = df.groupby('Player_ID')['Overall_Performance'].mean().nlargest(10)
    fig.add_trace(go.Bar(x=top_players.index, y=top_players.values, name='Top Performers', marker_color='#FF6B6B', showlegend=False), row=1, col=1)
    
    teams = df['Team'].unique()[:5]
    colors = px.colors.qualitative.Plotly
    stats_radar = ['Hits', 'Throws', 'Blocks', 'Dodges', 'Catches']
    for i, team in enumerate(teams):
        team_stats = df[df['Team'] == team][stats_radar].mean()
        fig.add_trace(go.Scatterpolar(r=team_stats.values, theta=stats_radar, fill='toself', name=team, line_color=colors[i]), row=1, col=2)
    
    if 'Player_Role' in df.columns:
        role_counts = df['Player_Role'].dropna().value_counts()
        if not role_counts.empty:
            fig.add_trace(go.Pie(labels=role_counts.index, values=role_counts.values, name="Roles", showlegend=False), row=2, col=1)
    
    player_summary = df.groupby('Player_ID').agg(Overall_Performance=('Overall_Performance', 'mean'), Win_Rate=('Win_Rate', 'first'), Consistency_Score=('Consistency_Score', 'first')).reset_index().dropna()
    fig.add_trace(go.Scatter(
        x=player_summary['Overall_Performance'], y=player_summary['Consistency_Score'], 
        mode='markers', text=player_summary['Player_ID'], 
        marker=dict(color='#4ECDC4', size=10), name='Performance vs Consistency',
        showlegend=False
    ), row=2, col=2)
    
    fig.update_layout(
        height=800, 
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1
        ),
        polar=dict(radialaxis=dict(visible=True, range=[0, df[stats_radar].max().max()]))
    )
    fig.update_xaxes(title_text="Average Performance (Skill) →", row=2, col=2)
    fig.update_yaxes(title_text="Consistency (Reliability) →", row=2, col=2)
    
    return fig

# In utils.py
def create_specialization_analysis(df):
    """
    Analyzes player specialization and returns a figure and top specialist DataFrames.
    """
    spec_stats = [
        'Hits', 'Throws', 'Dodges', 'Catches', 'Hit_Accuracy', 
        'Defensive_Efficiency', 'K/D_Ratio', 'Offensive_Rating', 'Defensive_Rating'
    ]
    
    player_avg_stats = df.groupby('Player_ID')[spec_stats].mean()
    league_avg = player_avg_stats.mean()
    specialization = player_avg_stats / (league_avg + 1e-6)
    
    # Create the heatmap figure
    top_specialized_players = specialization.std(axis=1).nlargest(20).index
    specialization_subset = specialization.loc[top_specialized_players]
    fig = px.imshow(
        specialization_subset, 
        text_auto=".2f", 
        aspect="auto", 
        color_continuous_scale='Viridis',
        labels=dict(x="Statistic", y="Player", color="Specialization Score (x League Avg)"),
        title="Player Specialization Heatmap (vs. League Average)"
    )
    fig.update_xaxes(side="top")
    
    # Get the top specialists DataFrames
    top_defensive = specialization.sort_values('Defensive_Rating', ascending=False).head(5)
    top_offensive = specialization.sort_values('Offensive_Rating', ascending=False).head(5)
    top_kd = specialization.sort_values('K/D_Ratio', ascending=False).head(5)
    
    return fig, top_defensive, top_offensive, top_kd

def generate_team_coaching_report(df, team_id):
    team_data = df[df['Team'] == team_id]
    if team_data.empty:
        return ["No data for this team."]
    team_avg_stats = team_data.mean(numeric_only=True)
    league_avg_stats = df[df['Team'] != team_id].mean(numeric_only=True)
    stats_to_compare = ['Hits', 'Throws', 'Catches', 'Dodges', 'Hit_Accuracy', 'Defensive_Efficiency', 'Overall_Performance', 'K/D_Ratio']
    weaknesses = {stat: (team_avg_stats.get(stat, 0) - league_avg_stats.get(stat, 0)) / (league_avg_stats.get(stat, 0) + 1e-6) for stat in stats_to_compare}
    biggest_weakness = min(weaknesses, key=weaknesses.get)
    report = [f"### Coaching Focus for {team_id}", f"**Biggest Team Weakness**: The team's **{biggest_weakness}** is the furthest below the league average."]
    advice_map = {
        'Hit_Accuracy': "🎯 **Team Focus**: Dedicate a session to throwing accuracy.",
        'Defensive_Efficiency': "🙌 **Team Focus**: Run drills that simulate 2-on-1 situations.",
        'Catches': "🛡️ **Team Focus**: Emphasize the value of catching.",
        'Dodges': "🏃 **Team Focus**: A full-team agility session could be beneficial.",
        'Overall_Performance': "📈 **Team Focus**: Go back to basics.",
        'K/D_Ratio': "⚡ **Team Focus**: The team needs to improve its elimination efficiency."
    }
    report.append(advice_map.get(biggest_weakness, "Focus on improving this area through targeted drills."))
    
    if 'Player_Role' in team_data.columns and not team_data['Player_Role'].isnull().all():
        has_catcher = any('Catcher' in str(role) for role in team_data['Player_Role'].unique())
        if not has_catcher:
            report.append("\n**Strategic Gap**: The team lacks a dedicated 'Catcher' type player.")
    return report

def generate_situational_insights(df):
    """
    Generate more advanced, AI-powered insights from the detailed situational data.
    """
    insights = []
    
    # Ensure data is numeric before calculating league average
    numeric_cols = df.select_dtypes(include=['number']).columns
    league_avg = df[numeric_cols].mean()

    # Insight 1: Find the biggest "Specialist"
    player_avg_stats = df.groupby('Player_ID')[numeric_cols].mean()
    specialization = player_avg_stats / (league_avg + 1e-6)
    
    # Find player with the highest single specialization score
    max_spec_player = specialization.max().idxmax()
    max_spec_value = specialization.max().max()
    player_with_max_spec = specialization[max_spec_player].idxmax()

    if max_spec_value > 2.5: # Only report if someone is > 2.5x league average
        insights.append(f"👑 **Extreme Specialist:** **{player_with_max_spec}** is a standout specialist in **{max_spec_player.replace('_', ' ')}**, performing at **{max_spec_value:.1f}x** the league average in that specific situation.")

    # Insight 2: Find players with the biggest performance gap
    df['offense_gap'] = df['Hits_Singles'] - df['Hits_Multi']
    df['defense_gap'] = df['Dodges_Singles'] - df['Dodges_Multi']
    
    biggest_offense_gap_player = df.loc[df['offense_gap'].idxmax()]
    if biggest_offense_gap_player['offense_gap'] > 5: # If the gap is more than 5 hits
        insights.append(f"🎯 **Singles Dominator:** **{biggest_offense_gap_player['Player_ID']}** is significantly more effective in single-ball situations, with **{biggest_offense_gap_player['offense_gap']:.0f}** more hits in Singles than Multi-ball.")

    biggest_defense_gap_player = df.loc[df['defense_gap'].idxmax()]
    if biggest_defense_gap_player['defense_gap'] > 5: # If the gap is more than 5 dodges
        insights.append(f"🏃 **Evasive Specialist:** **{biggest_defense_gap_player['Player_ID']}** relies heavily on dodging in single-ball situations, with **{biggest_defense_gap_player['defense_gap']:.0f}** more dodges in Singles than Multi-ball.")

    # Insight 3: Team-level Tactical Insight
    team_counter_hits = df.groupby('Team')['Hits_Counters'].sum()
    team_total_hits = df.groupby('Team')['Hits'].sum()
    team_counter_ratio = (team_counter_hits / team_total_hits).dropna()
    
    if not team_counter_ratio.empty:
        top_counter_team = team_counter_ratio.idxmax()
        top_ratio = team_counter_ratio.max()
        if top_ratio > 0.4: # If more than 40% of hits are from counters
            insights.append(f"💥 **Counter-Attack Kings:** **{top_counter_team}** excels at turning defense into offense. A massive **{top_ratio:.1%}** of their total hits come from counter-attacks.")

    if not insights:
        insights.append("✅ **Balanced Dataset:** No extreme outliers or specialists were identified. The teams and players in this dataset show relatively balanced performance across different situations.")

    return insights
