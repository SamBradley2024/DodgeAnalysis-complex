import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
import numpy as np
import warnings
import gspread
from google.oauth2.service_account import Credentials
import os

warnings.filterwarnings('ignore')

# --- 1. COMPLETELY REWRITTEN DATA PROCESSING FUNCTION ---
def load_and_process_uploaded_csvs(uploaded_files):
    """
    This function is completely rewritten to handle the specific matrix format
    of the 'Dodgeball App Data - Meet X.csv' files.
    """
    all_games_data = []

    for uploaded_file in uploaded_files:
        match_id = os.path.splitext(uploaded_file.name)[0].split(' - ')[-1]
        raw_df = pd.read_csv(uploaded_file, header=None, dtype=str).fillna('')

        # --- Logic to split files with multiple games ---
        empty_cols = (raw_df == '').all()
        split_indices = [i for i, is_empty in enumerate(empty_cols) if is_empty]
        game_chunks = []
        start_col = 0
        split_indices.append(raw_df.shape[1])

        for split_index in split_indices:
            chunk = raw_df.iloc[:, start_col:split_index].copy()
            chunk.dropna(how='all', axis=1, inplace=True)
            chunk.dropna(how='all', axis=0, inplace=True)
            if not chunk.empty:
                game_chunks.append(chunk)
            start_col = split_index + 1

        # --- Process each game found in the file ---
        for game_df in game_chunks:
            # First row is header: opponent in first cell, players in the rest
            opponent_name = game_df.iloc[0, 0].replace('vs ', '').strip()
            player_names = game_df.iloc[0, 1:]
            
            # Get stats (all rows after header)
            stats_df = game_df.iloc[1:]
            stat_names = stats_df.iloc[:, 0]
            stat_values = stats_df.iloc[:, 1:]

            # Create a new DataFrame for this game
            processed_game = pd.DataFrame(stat_values.values, columns=player_names, index=stat_names)
            
            # Transpose so players are rows and stats are columns
            processed_game = processed_game.T
            
            # Reset index to turn player names into a column
            processed_game = processed_game.reset_index().rename(columns={'index': 'Player'})

            # Add the identifiers we created
            processed_game['Opponent'] = opponent_name
            processed_game['Match_ID'] = match_id
            processed_game['Game_ID'] = f"{match_id} vs {opponent_name}"
            
            all_games_data.append(processed_game)

    if not all_games_data:
        st.error("Could not extract any game data from the uploaded file(s).")
        return pd.DataFrame()

    master_df = pd.concat(all_games_data, ignore_index=True)

    # --- Data Cleaning and Type Conversion ---
    for col in master_df.columns:
        if col not in ['Player', 'Opponent', 'Game_ID', 'Match_ID']:
            master_df[col] = pd.to_numeric(
                master_df[col].astype(str).str.replace('%', '', regex=False),
                errors='coerce'
            )

    # --- Column Name Mapping ---
    # Map the stat names from the CSV to the names the app expects
    column_mapping = {
        'Player': 'Player_ID',
        'Overall Hits': 'Total_Hits',
        'Overall Throws': 'Total_Throws',
        'Catches made': 'Total_Catches',
        'Dodges (overall)': 'Total_Dodges',
        'Blocks (overall)': 'Total_Blocks',
        'Out (overall hits)': 'Total_Hit_Out',
        'Out (caught)': 'Total_Caught_Out',
        'Sets Played': 'Sets_Played',
    }
    
    processed_df = master_df.rename(columns=column_mapping)
    
    # Add required columns that don't exist in the raw data
    processed_df['Team'] = "Notts Women's 1s"
    processed_df['Game_Outcome'] = 'Unknown' # Outcome is not in the stats sheet

    # Ensure all required numeric columns exist, fill with 0 if not
    expected_cols = list(column_mapping.values())
    for col in expected_cols:
         if col not in processed_df.columns and col != 'Player_ID':
             processed_df[col] = 0

    return processed_df


# --- 2. STYLING AND UI HELPERS ---
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
            color: #313131;
        }
    </style>
    """

def styled_metric(label, value):
    """Creates a styled metric box."""
    st.markdown(f'<div class="metric-container"><h4>{label}</h4><h3>{value}</h3></div>', unsafe_allow_html=True)


# --- 3. DATA LOADING AND INITIALIZATION ---
@st.cache_data(ttl=600)
def load_from_google_sheet(worksheet_name):
    """Loads data from the specified Google Sheet worksheet."""
    try:
        scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
        creds_dict = {
            "type": st.secrets["gcp_service_account"]["type"], "project_id": st.secrets["gcp_service_account"]["project_id"],
            "private_key_id": st.secrets["gcp_service_account"]["private_key_id"], "private_key": st.secrets["gcp_service_account"]["private_key"],
            "client_email": st.secrets["gcp_service_account"]["client_email"], "client_id": st.secrets["gcp_service_account"]["client_id"],
            "auth_uri": st.secrets["gcp_service_account"]["auth_uri"], "token_uri": st.secrets["gcp_service_account"]["token_uri"],
            "auth_provider_x509_cert_url": st.secrets["gcp_service_account"]["auth_provider_x509_cert_url"],
            "client_x509_cert_url": st.secrets["gcp_service_account"]["client_x509_cert_url"]
        }
        creds = Credentials.from_service_account_info(creds_dict, scopes=scope)
        client = gspread.authorize(creds)
        spreadsheet = client.open("Dodgeball App Data")
        worksheet = spreadsheet.worksheet(worksheet_name)
        data = worksheet.get_all_records()
        return pd.DataFrame(data)
    except Exception as e:
        st.error(f"Failed to load from Google Sheets: {e}")
        return None

@st.cache_data
def get_worksheet_names():
    # This function remains unchanged
    pass # Keep your existing function here

@st.cache_resource
def train_models(_df):
    # This function remains unchanged
    pass # Keep your existing function here

@st.cache_data
def enhance_data(_df):
    """
    Performs feature engineering. This is now robust to the new data.
    """
    df = _df.copy()
    
    # Recalculate metrics to ensure consistency
    df['Hit_Accuracy'] = (df['Total_Hits'] / df['Total_Throws']).fillna(0)
    df['KD_Ratio'] = (df['Total_Hits'] / df['Total_Hit_Out']).replace([np.inf, -np.inf], 0).fillna(0)
    df['Catch_Success_Rate'] = (df['Total_Catches'] / (df['Total_Catches'] + df['Total_Caught_Out'])).fillna(0)
    df['Offensive_Rating'] = (df['Total_Hits'] * 0.7) + (df['Total_Throws'] * 0.3)
    df['Defensive_Rating'] = (df['Total_Catches'] * 0.5) + (df['Total_Dodges'] * 0.3) + (df['Total_Blocks'] * 0.2)
    df['Overall_Performance'] = (df['Offensive_Rating'] * 0.6) + (df['Defensive_Rating'] * 0.4)

    # Calculate player averages across all their games
    player_avg = df.groupby('Player_ID').agg(
        Avg_Hits=('Total_Hits', 'mean'), Avg_Throws=('Total_Throws', 'mean'),
        Avg_Catches=('Total_Catches', 'mean'), Avg_Dodges=('Total_Dodges', 'mean'),
        Avg_Blocks=('Total_Blocks', 'mean'), Avg_Hit_Out=('Total_Hit_Out', 'mean'),
        Avg_Caught_Out=('Total_Caught_Out', 'mean'), Avg_Hit_Accuracy=('Hit_Accuracy', 'mean'),
        Avg_KD_Ratio=('KD_Ratio', 'mean'), Avg_Catch_Success_Rate=('Catch_Success_Rate', 'mean'),
        Avg_Performance=('Overall_Performance', 'mean')
    ).reset_index()

    df = pd.merge(df, player_avg, on='Player_ID', how='left')
    
    df['Match_Order'] = df['Match_ID'].astype('category').cat.codes
    stamina_data = df.groupby('Player_ID')[['Match_Order', 'Overall_Performance']].corr(numeric_only=True).unstack().iloc[:, 1].rename('Stamina_Trend')
    df = df.merge(stamina_data, on='Player_ID', how='left')

    return df

def initialize_app(raw_df, source_name):
    """Main initialization function."""
    with st.spinner("Processing data and training AI models..."):
        st.session_state.df_raw = raw_df
        st.session_state.df_enhanced = enhance_data(raw_df.copy())
        # st.session_state.models = train_models(st.session_state.df_enhanced) # Optional: uncomment if you need the ML models
        st.session_state.source_name = source_name
        st.session_state.data_loaded = True
        st.success(f"Successfully loaded and processed data from {source_name}!")


# --- 4. CORE ANALYSIS & VISUALIZATION FUNCTIONS ---
# All of your original functions for creating charts and generating insights.
def create_game_level_features(df):
    """Transforms player-level data into game-level data for win prediction."""
    team_game_stats = df.groupby(['Game_ID', 'Team']).agg(
        Avg_Performance=('Overall_Performance', 'mean'),
        Avg_KD_Ratio=('KD_Ratio', 'mean'),
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

def create_improvement_chart(df, player_id, metric_to_plot='Overall_Performance'):
    """Creates a line chart showing a player's performance over time with a trendline."""
    player_df = df[df['Player_ID'] == player_id].copy()
    
    # Calculate average of the chosen metric per match
    match_performance = player_df.groupby('Match_ID')[metric_to_plot].mean().reset_index()
    
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
        y=metric_to_plot,
        title=f'{player_id} - {metric_name_formatted} Trend Across Matches',
        labels={"match_num": "Match", metric_to_plot: f"Average {metric_name_formatted}"},
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

    # Insight 1: Top Performer
    top_performer = df.groupby('Player_ID')['Overall_Performance'].mean().idxmax()
    top_score = df.groupby('Player_ID')['Overall_Performance'].mean().max()
    insights.append(f"🏆 **Top Performer**: {top_performer} leads the league with an average performance score of {top_score:.2f}.")

    # Insight 2: Key to Success
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
            insights.append(f"📈 **Key to Success**: In this dataset, **{top_corr_skill.replace('_', ' ')}** has the strongest correlation with a player's Overall Performance score.")

    # Insight 3: Team Statistical Gaps
    team_summary = df.groupby('Team').agg(
        Avg_Performance=('Avg_Performance', 'first')
    ).dropna()

    if not team_summary.empty and len(team_summary) > 1:
        stats_to_check = ['Avg_Hit_Accuracy', 'Avg_KD_Ratio', 'Avg_Dodges', 'Avg_Catches']
        existing_stats = [stat for stat in stats_to_check if stat in df.columns]
        
        if existing_stats:
            league_avg = df[existing_stats].mean()

            for team_name in team_summary.index:
                team_stats = df[df['Team'] == team_name][existing_stats].mean()
                comparison = (team_stats - league_avg) / league_avg
                
                if not comparison.empty:
                    weakest_stat = comparison.idxmin()
                    weakness_value = comparison.min()
                    
                    if weakness_value < -0.15:
                        weakness_stat_clean = weakest_stat.replace('Avg_', '').replace('_', ' ')
                        insights.append(f"💡 **Coaching Focus for {team_name}**: Their biggest statistical weakness is in **{weakness_stat_clean}**, which is {abs(weakness_value):.0%} below the league average.")

    # Insight 4: Stamina Analysis
    if 'Stamina_Trend' in df.columns:
        stamina_data = df.groupby('Player_ID')['Stamina_Trend'].first().dropna()
        if not stamina_data.empty:
            worst_stamina_player = stamina_data.idxmin()
            worst_stamina_score = stamina_data.min()
            if worst_stamina_score < -0.3:
                insights.append(f"🏃 **Stamina Watch**: **{worst_stamina_player}** shows a tendency to fade, as their performance drops significantly in later games within a match.")

            best_stamina_player = stamina_data.idxmax()
            best_stamina_score = stamina_data.max()
            if best_stamina_score > 0.3:
                insights.append(f"⚡ **Strong Finisher**: **{best_stamina_player}** is a clutch player who consistently improves their performance as a match progresses.")
                
    return insights