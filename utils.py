import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
import warnings
import gspread
from google.oauth2.service_account import Credentials

warnings.filterwarnings('ignore')


# --- Styling and UI Helpers ---
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
    st.markdown('<div class="metric-container">', unsafe_allow_html=True)
    st.metric(label, value, help=help_text)
    st.markdown('</div>', unsafe_allow_html=True)

# --- Data Loading and Processing Functions ---

@st.cache_data(ttl=300)
def load_from_google_sheet(worksheet_name):
    """Loads a DataFrame from a specific Google Sheet worksheet."""
    try:
        scopes = ["https://www.googleapis.com/auth/spreadsheets", "https://www.googleapis.com/auth/drive.readonly"]
        creds = Credentials.from_service_account_info(st.secrets["gcp_service_account"], scopes=scopes)
        client = gspread.authorize(creds)
        sheet = client.open("Dodgeball App Data").worksheet(worksheet_name)
        data = sheet.get_all_records()
        if not data:
            st.warning(f"Worksheet '{worksheet_name}' is empty or has no data.")
            return None
        return pd.DataFrame(data)
    except Exception as e:
        st.error(f"Error reading from Google Sheets: {e}")
        return None

@st.cache_resource(ttl=600)  # Cache for 10 minutes
def get_gspread_client():
    """Initializes and returns the gspread client."""
    scope = [
        "https://spreadsheets.google.com/feeds",
        "https://www.googleapis.com/auth/spreadsheets",
        "https://www.googleapis.com/auth/drive.file",
        "https://www.googleapis.com/auth/drive",
    ]
    # Use Streamlit's secrets management for the credentials
    creds_dict = st.secrets["gcp_service_account"]
    creds = Credentials.from_service_account_info(creds_dict, scopes=scope)
    client = gspread.authorize(creds)
    return client

def get_worksheet_names():
    """Fetches the names of all worksheets in the 'Dodgeball App Data' Google Sheet."""
    try:
        client = get_gspread_client()
        spreadsheet = client.open("Dodgeball App Data")
        return [sheet.title for sheet in spreadsheet.worksheets()]
    except Exception as e:
        st.error(f"Could not connect to Google Sheets. Please ensure you have set up your secrets correctly. Error: {e}")
        return []

def load_and_process_google_sheet(worksheet_name):
    """
    Loads and processes a worksheet from Google Sheets with the new, pivoted format.
    This is a more robust version that builds the DataFrame directly.
    """
    try:
        client = get_gspread_client()
        worksheet = client.open("Dodgeball App Data").worksheet(worksheet_name)
        
        # Get all values as a list of lists
        data = worksheet.get_all_values()
        
        if not data or len(data) < 2:
            st.error(f"Worksheet '{worksheet_name}' is empty or has no data rows.")
            return None

        # The first row contains headers (team info and player names)
        header_row = data[0]
        player_names = header_row[1:]
        
        # Subsequent rows contain the metrics and values
        metric_rows = data[1:]
        
        # Build a dictionary to construct the DataFrame
        # This is a more stable method than transposing
        processed_data = {'Player_ID': player_names}
        for row in metric_rows:
            metric_name = row[0]
            metric_values = row[1:]
            # Ensure each metric has a value for each player
            if len(metric_values) == len(player_names):
                processed_data[metric_name] = metric_values
        
        df = pd.DataFrame(processed_data)

        # --- (The rest of the logic remains the same) ---
        team_name = header_row[0].split(' vs ')[0]
        df['Team'] = team_name
        df['Match_ID'] = 'M1'
        df['Game_ID'] = 'G1'
        df['Game_Outcome'] = 'Win'  # Default value

        df = df.rename(columns={
            'Overall Hits': 'Hits', 'Overall Throws': 'Throws', 'Catches made': 'Catches',
            'Overall Outs': 'Times_Eliminated', 'Out (caught)': 'Caught_Out', 'Out (overall hits)': 'Hit_Out'
        })
        
        df['Dodges'] = 0
        df['Blocks'] = 0
        
        return df
    except Exception as e:
        st.error(f"Error processing the Google Sheet '{worksheet_name}': {e}")
        return None

def load_and_process_custom_csv(uploaded_file):
    """
    Loads and processes a custom-formatted CSV file into a tidy DataFrame.
    This version is more robust and mirrors the Google Sheet logic.
    """
    try:
        # Read the raw data without assuming a header or index
        df_raw = pd.read_csv(uploaded_file, header=None)
        
        if df_raw.empty or len(df_raw) < 2:
            st.error("The uploaded CSV file is empty or has no data rows.")
            return None

        # The first row contains headers (team info and player names)
        header_row = df_raw.iloc[0].tolist()
        player_names = header_row[1:]
        
        # Subsequent rows contain the metrics and values
        metric_rows = df_raw.iloc[1:].values.tolist()

        # Build a dictionary to construct the DataFrame
        processed_data = {'Player_ID': player_names}
        for row in metric_rows:
            metric_name = row[0]
            metric_values = row[1:]
            if len(metric_values) == len(player_names):
                processed_data[metric_name] = metric_values
        
        df = pd.DataFrame(processed_data)

        # --- (The rest of the logic is the same as the Google Sheet function) ---
        team_name = header_row[0].split(' vs ')[0]
        df['Team'] = team_name
        df['Match_ID'] = 'M1'
        df['Game_ID'] = 'G1'
        df['Game_Outcome'] = 'Win'  # Default value

        df = df.rename(columns={
            'Overall Hits': 'Hits', 'Overall Throws': 'Throws', 'Catches made': 'Catches',
            'Overall Outs': 'Times_Eliminated', 'Out (caught)': 'Caught_Out', 'Out (overall hits)': 'Hit_Out'
        })
        
        df['Dodges'] = 0
        df['Blocks'] = 0
        
        return df
    except Exception as e:
        st.error(f"Error processing the CSV file: {e}")
        return None


def enhance_dataframe(df):
    """Takes a raw dataframe and adds all calculated metrics and features."""
    # --- MODIFIED ---
    # The new required columns after processing the custom CSV
    required_cols = ['Match_ID', 'Player_ID', 'Team', 'Game_ID', 'Game_Outcome', 
                     'Hits', 'Throws', 'Catches', 'Times_Eliminated', 
                     'Caught_Out', 'Hit_Out', 'Dodges', 'Blocks']
    
    if not all(col in df.columns for col in required_cols):
        st.error("The data is missing required columns. Please ensure it has: " + ", ".join(required_cols))
        return None

    # Convert numeric columns to numeric types
    numeric_cols = ['Hits', 'Throws', 'Catches', 'Times_Eliminated', 'Caught_Out', 'Hit_Out', 'Dodges', 'Blocks']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    # Calculate metrics
    df['K/D_Ratio'] = df['Hits'] / df['Times_Eliminated'].replace(0, 1)
    df['Net_Impact'] = (df['Hits'] + df['Catches']) - df['Times_Eliminated']
    df['Hit_Accuracy'] = np.where(df['Throws'] > 0, df['Hits'] / df['Throws'], 0)
    df['Defensive_Efficiency'] = np.where((df['Catches'] + df['Dodges'] + df['Hit_Out']) > 0, (df['Catches'] + df['Dodges']) / (df['Catches'] + df['Dodges'] + df['Hit_Out']), 0)
    df['Offensive_Rating'] = (df['Hits'] * 2 + df['Throws'] * 0.5) / (df['Throws'] + 1)
    df['Defensive_Rating'] = (df['Dodges'] + df['Catches'] * 2) / 3
    df['Overall_Performance'] = (df['Offensive_Rating'] * 0.35 + df['Defensive_Rating'] * 0.35 + df['K/D_Ratio'] * 0.15 + df['Net_Impact'] * 0.05 + df['Hit_Accuracy'] * 0.05 + df['Defensive_Efficiency'] * 0.05)
    df['Game_Impact'] = np.where(df['Game_Outcome'] == 'Win', df['Overall_Performance'] * 1.2, df['Overall_Performance'] * 0.8)


@st.cache_resource
def train_advanced_models(_df):
    """Trains ML models and generates player roles."""
    df = _df.copy()
    models = {}
    
    role_features = ['Hits', 'Throws', 'Dodges', 'Catches', 'Hit_Accuracy', 'Defensive_Efficiency', 'Offensive_Rating', 'Defensive_Rating', 'K/D_Ratio']
    df_role_features = df[role_features].dropna()

    if df_role_features.empty or len(df_role_features) < 4:
        st.warning("Not enough data to create player roles.")
        df['Player_Role'] = 'N/A'
        return df, models

    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(df_role_features)
    kmeans = KMeans(n_clusters=4, random_state=42, n_init='auto')
    
    df.loc[df_role_features.index, 'Role_Cluster'] = kmeans.fit_predict(scaled_features)
    
    cluster_centers_unscaled = scaler.inverse_transform(kmeans.cluster_centers_)
    league_average_stats = df_role_features.mean()
    role_names = []
    
    name_map = {
        'Hits': 'High Hits', 'Throws': 'High Volume Thrower', 'Dodges': 'Evasive',
        'Catches': 'Catcher', 'Hit_Accuracy': 'Precision Player', 'Defensive_Efficiency': 'Efficient Defender',
        'Offensive_Rating': 'Offensive', 'Defensive_Rating': 'Defensive', 'K/D_Ratio': 'High K/D Player'
    }

    for i in range(cluster_centers_unscaled.shape[0]):
        center_stats = pd.Series(cluster_centers_unscaled[i], index=role_features)
        specialization_scores = (center_stats - league_average_stats) / (league_average_stats + 1e-6)
        top_specializations = specialization_scores.nlargest(2)
        primary_spec_name = name_map.get(top_specializations.index[0], top_specializations.index[0])
        secondary_spec_name = name_map.get(top_specializations.index[1], top_specializations.index[1])
        base_role_name = f"{primary_spec_name}-{secondary_spec_name} Hybrid"
        final_role_name = base_role_name
        counter = 1
        while final_role_name in role_names:
            counter += 1
            final_role_name = f"{base_role_name} ({counter})"
        role_names.append(final_role_name)
    
    role_mapping = {float(i): role_names[i] for i in range(len(role_names))}
    df['Player_Role'] = df['Role_Cluster'].map(role_mapping).fillna('Generalist')
    
    models['role_model'] = (kmeans, scaler, role_mapping, role_names)
    
    return df, models

def initialize_app(df, source_name):
    """Initializes the app by processing data and training models."""
    with st.spinner(f"Processing data from '{source_name}' and training models..."):
        df_enhanced = enhance_dataframe(df.copy())
        if df_enhanced is not None:
            df_trained, models = train_advanced_models(df_enhanced)
            
            game_level_df = create_game_level_features(df_trained)
            win_model, win_model_features = train_win_prediction_model(game_level_df)
            if win_model:
                models['win_predictor'] = (win_model, win_model_features)
            
            st.session_state.df_enhanced = df_trained
            st.session_state.models = models
            st.session_state.data_loaded = True
            st.session_state.source_name = source_name

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


# --- Visualization Functions ---
# In utils.py

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

def generate_player_coaching_report(df, player_id):
    player_data = df[df['Player_ID'] == player_id]
    if player_data.empty:
        return ["No data for this player."], None

    player_avg_stats = player_data.mean(numeric_only=True)
    
    if 'Player_Role' not in player_data.columns or player_data['Player_Role'].isnull().all():
        player_role = "N/A"
        role_avg_stats = pd.Series()
    else:
        player_role = player_data['Player_Role'].iloc[0]
        role_avg_stats = df[df['Player_Role'] == player_role].mean(numeric_only=True)

    league_avg_stats = df.mean(numeric_only=True)
    stats_to_compare = ['Hits', 'Throws', 'Catches', 'Dodges', 'Hit_Accuracy', 'Defensive_Efficiency', 'K/D_Ratio']
    
    role_weaknesses = {}
    if not role_avg_stats.empty:
        role_weaknesses = {stat: (player_avg_stats.get(stat, 0) - role_avg_stats.get(stat, 0)) / (role_avg_stats.get(stat, 0) + 1e-6) for stat in stats_to_compare}
        role_weaknesses = {k: v for k, v in role_weaknesses.items() if v < -0.1}

    overall_weaknesses = {stat: (player_avg_stats.get(stat, 0) - league_avg_stats.get(stat, 0)) / (league_avg_stats.get(stat, 0) + 1e-6) for stat in stats_to_compare}
    overall_weaknesses = {k: v for k, v in overall_weaknesses.items() if v < -0.1}

    report = [f"### Coaching Focus for {player_id} ({player_role})"]
    advice_map = {
        'Hit_Accuracy': "🎯 **Suggestion**: Focus on throwing drills.",
        'Defensive_Efficiency': "🙌 **Suggestion**: Improve decision-making when targeted.",
        'Catches': "🛡️ **Suggestion**: Improve positioning and anticipation.",
        'Dodges': "🏃 **Suggestion**: Enhance agility and footwork.",
        'Hits': "💥 **Suggestion**: Be more aggressive offensively.",
        'K/D_Ratio': "⚡ **Suggestion**: Focus on survivability."
    }
    if not role_weaknesses and not overall_weaknesses:
        report.append("✅ **Well-Rounded Performer**: This player is performing at or above average.")
    if role_weaknesses:
        stat = min(role_weaknesses, key=role_weaknesses.get)
        report.append(f"**Role-Specific Weakness**: **{stat}**.")
        report.append(advice_map.get(stat))
    if overall_weaknesses:
        stat = min(overall_weaknesses, key=overall_weaknesses.get)
        report.append(f"**Overall Weakness**: **{stat}**.")
        if not role_weaknesses or stat != min(role_weaknesses, key=role_weaknesses.get):
             report.append(advice_map.get(stat))
    
    fig = go.Figure()
    fig.add_trace(go.Bar(name=f'{player_id} (You)', x=stats_to_compare, y=[player_avg_stats.get(s, 0) for s in stats_to_compare], marker_color='#FF6B6B'))
    if not role_avg_stats.empty:
        fig.add_trace(go.Bar(name='Role Average', x=stats_to_compare, y=[role_avg_stats.get(s, 0) for s in stats_to_compare], marker_color='#4ECDC4'))
    fig.add_trace(go.Bar(name='League Average', x=stats_to_compare, y=[league_avg_stats.get(s, 0) for s in stats_to_compare], marker_color='#45B7D1'))
    
    fig.update_layout(barmode='group', title_text='Performance Comparison', xaxis_title="Statistic", yaxis_title="Average Value")
    return report, fig

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

def generate_insights(df, models):
    """Generate more advanced, AI-powered insights from the data."""
    insights = []

    # Insight 1: Top Performer (Unbolded)
    top_performer = df.groupby('Player_ID')['Overall_Performance'].mean().idxmax()
    top_score = df.groupby('Player_ID')['Overall_Performance'].mean().max()
    insights.append(f"🏆 Top Performer: {top_performer} leads the league with an average performance score of {top_score:.2f}.")

    # Insight 2: Key to Success (Unbolded)
    corr_cols = [
        'Hits', 'Throws', 'Catches', 'Dodges', 'Blocks', 'Hit_Accuracy', 
        'K/D_Ratio', 'Net_Impact', 'Defensive_Efficiency', 
        'Offensive_Rating', 'Defensive_Rating', 'Overall_Performance'
    ]
    existing_corr_cols = [col for col in corr_cols if col in df.columns]
    if 'Overall_Performance' in existing_corr_cols:
        performance_corr = df[existing_corr_cols].corr()['Overall_Performance'].abs().sort_values(ascending=False)
        if len(performance_corr) > 1:
            top_corr_skill = performance_corr.index[1]
            insights.append(f"📈 Key to Success: In this dataset, {top_corr_skill.replace('_', ' ')} has the strongest correlation with a player's Overall Performance score.")

    # --- UPDATED: Insight 3 now analyzes statistical gaps for ALL teams ---
    team_summary = df.groupby('Team').agg(
        Avg_Performance=('Avg_Performance', 'first')
    ).dropna()

    if not team_summary.empty and len(team_summary) > 1:
        stats_to_check = ['Avg_Hit_Accuracy', 'Avg_KD_Ratio', 'Avg_Dodges', 'Avg_Catches']
        existing_stats = [stat for stat in stats_to_check if stat in df.columns]
        
        if existing_stats:
            league_avg = df[existing_stats].mean()

            # Loop through every team to find their biggest weakness
            for team_name in team_summary.index:
                team_stats = df[df['Team'] == team_name][existing_stats].mean()
                comparison = (team_stats - league_avg) / league_avg
                
                if not comparison.empty:
                    weakest_stat = comparison.idxmin()
                    weakness_value = comparison.min()
                    
                    # Only report the insight if the weakness is significant (e.g., >15% below average)
                    if weakness_value < -0.15:
                        weakness_stat_clean = weakest_stat.replace('Avg_', '').replace('_', ' ')
                        insights.append(f"💡 Coaching Focus for {team_name}: Their biggest statistical weakness is in {weakness_stat_clean}, which is {abs(weakness_value):.0%} below the league average.")

    # UPDATED: Insight 4 now analyzes stamina
    if 'Stamina_Trend' in df.columns:
        stamina_data = df.groupby('Player_ID')['Stamina_Trend'].first().dropna()
        if not stamina_data.empty:
            # Find player who fades the most (most negative correlation)
            worst_stamina_player = stamina_data.idxmin()
            worst_stamina_score = stamina_data.min()
            if worst_stamina_score < -0.3:
                insights.append(f"🏃 Stamina Watch: {worst_stamina_player} shows a tendency to fade, as their performance drops significantly in later games within a match.")

            # Find player who gets stronger (most positive correlation)
            best_stamina_player = stamina_data.idxmax()
            best_stamina_score = stamina_data.max()
            if best_stamina_score > 0.3:
                insights.append(f"⚡ Strong Finisher: {best_stamina_player} is a clutch player who consistently improves their performance as a match progresses.")
            
    return insights