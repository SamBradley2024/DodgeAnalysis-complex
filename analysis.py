import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import warnings
from plotly.subplots import make_subplots
import plotly.graph_objects as go
warnings.filterwarnings('ignore')

# --- Page Configuration ---
st.set_page_config(
    page_title="Advanced Dodgeball Analytics Dashboard",
    page_icon="🤾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #FF6B6B, #4ECDC4);
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        color: white;
        text-align: center;
    }
    .metric-container {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #4ECDC4;
        margin: 0.5rem 0;
    }
    .insight-box {
        background: #e8f4fd;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #1f77b4;
        margin: 1rem 0;
    }
    .warning-box {
        background: #fff3cd;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #ffc107;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# --- Enhanced Data Loading and Feature Engineering ---
@st.cache_data
def load_and_enhance_data(filepath):
    """Enhanced data loading with comprehensive feature engineering."""
    if not os.path.exists(filepath):
        st.error(f"Error: The file '{filepath}' was not found.")
        st.info("Please create a file named `dodgeball_data.csv` in the same directory.")
        return None
    
    df = pd.read_csv(filepath)
    
    # Basic efficiency metrics
    df['Hit_Accuracy'] = np.where(df['Throws'] > 0, df['Hits'] / df['Throws'], 0)
    df['Defensive_Efficiency'] = np.where((df['Catches'] + df['Dodges'] + df['Hit_Out']) > 0, 
                                          (df['Catches'] + df['Dodges']) / (df['Catches'] + df['Dodges'] + df['Hit_Out']), 0)
    
    # Advanced performance metrics
    df['Offensive_Rating'] = (df['Hits'] * 2 + df['Throws'] * 0.5) / (df['Throws'] + 1)
    df['Defensive_Rating'] = (df['Dodges'] + df['Catches'] * 2) / 3 
    df['Elimination_Rate'] = df['Hit_Out'] + df['Caught_Out']
    df['Survival_Score'] = 10 - df['Elimination_Rate']
    
    # Composite performance score (rebalanced weights)
    df['Overall_Performance'] = (
        df['Offensive_Rating'] * 0.4 + 
        df['Defensive_Rating'] * 0.4 + 
        df['Hit_Accuracy'] * 0.1 + 
        df['Defensive_Efficiency'] * 0.1
    )
    
    # Game impact metrics
    df['Game_Impact'] = np.where(df['Game_Outcome'] == 'Win', 
                                 df['Overall_Performance'] * 1.2, 
                                 df['Overall_Performance'] * 0.8)
    
    # Consistency metrics
    player_stats = df.groupby('Player_ID').agg({
        'Overall_Performance': ['mean', 'std'],
        'Hit_Accuracy': ['mean', 'std'],
        'Game_Outcome': lambda x: (x == 'Win').mean()
    }).round(3)
    
    player_stats.columns = ['Avg_Performance', 'Performance_Consistency', 
                            'Avg_Hit_Accuracy', 'Hit_Accuracy_Consistency', 
                            'Win_Rate']
    
    player_stats['Consistency_Score'] = 1 / (player_stats['Performance_Consistency'] + 0.01)
    
    df = df.merge(player_stats, left_on='Player_ID', right_index=True, how='left')
    
    return df

# --- Advanced ML Models ---
@st.cache_resource
def train_advanced_models(df):
    """Train multiple ML models for different predictions."""
    models = {}
    
    # Player Role Classification
    role_features = ['Hits', 'Throws', 'Dodges', 'Catches', 
                     'Hit_Accuracy', 'Defensive_Efficiency', 'Offensive_Rating', 'Defensive_Rating']
    
    scaler = StandardScaler()
    df_role_features = df[role_features].dropna()
    scaled_features = scaler.fit_transform(df_role_features)
    
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=20)
    df.loc[df_role_features.index, 'Role_Cluster'] = kmeans.fit_predict(scaled_features)
    
    cluster_centers_unscaled = scaler.inverse_transform(kmeans.cluster_centers_)
    league_average_stats = df_role_features.mean()
    
    role_names = []
    name_map = {
        'Hits': 'Striker', 'Throws': 'Thrower',
        'Dodges': 'Evader', 'Catches': 'Catcher', 'Hit_Accuracy': 'Accurate',
        'Defensive_Efficiency': 'Efficient', 'Offensive_Rating': 'Offensive', 'Defensive_Rating': 'Defensive'
    }

    for i in range(cluster_centers_unscaled.shape[0]):
        center_stats = pd.Series(cluster_centers_unscaled[i], index=role_features)
        
        specialization_scores = (center_stats - league_average_stats) / (league_average_stats + 1e-6)
        top_specializations = specialization_scores.nlargest(2)
        
        primary_spec_name = top_specializations.index[0]
        secondary_spec_name = top_specializations.index[1]
        
        role_name_1 = name_map.get(primary_spec_name, primary_spec_name)
        role_name_2 = name_map.get(secondary_spec_name, secondary_spec_name)
        
        base_role_name = f"{role_name_1}-{role_name_2} Hybrid"

        specialization_std = specialization_scores.std()
        if specialization_std < 0.5:
            role_name = f"Balanced {role_name_1}"
        elif specialization_std > 1.5:
            role_name = f"Focused {role_name_1}"
        else:
            role_name = base_role_name

        final_role_name = role_name
        counter = 1
        while final_role_name in role_names:
            counter += 1
            final_role_name = f"{role_name} ({counter})"
            
        role_names.append(final_role_name)

    role_mapping = {i: role_names[i] for i in range(len(role_names))}
    df['Player_Role'] = df['Role_Cluster'].map(role_mapping)
    models['role_model'] = (kmeans, scaler, role_mapping, role_names)
    
    # Game Outcome Prediction
    outcome_features = ['Hits', 'Throws', 'Dodges', 'Catches', 
                        'Overall_Performance', 'Offensive_Rating', 'Defensive_Rating']
    
    outcome_df = df.dropna(subset=outcome_features + ['Game_Outcome'])
    le = LabelEncoder()
    y = le.fit_transform(outcome_df['Game_Outcome'])
    X = outcome_df[outcome_features]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_classifier.fit(X_train, y_train)
    
    accuracy = accuracy_score(y_test, rf_classifier.predict(X_test))
    models['outcome_model'] = (rf_classifier, le, accuracy, outcome_features)
    
    # Performance Prediction
    perf_features = ['Hits', 'Throws', 'Dodges', 'Catches']
    perf_df = df.dropna(subset=perf_features + ['Overall_Performance'])
    gb_regressor = GradientBoostingRegressor(n_estimators=100, random_state=42)
    gb_regressor.fit(perf_df[perf_features], perf_df['Overall_Performance'])
    models['performance_model'] = (gb_regressor, perf_features)
    
    return df, models

# --- Enhanced Visualization Functions ---
def create_player_dashboard(df, player_id):
    """Create comprehensive player dashboard with multiple visualizations."""
    player_data = df[df['Player_ID'] == player_id]
    
    if player_data.empty:
        st.error(f"No data found for player {player_id}")
        return
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Performance Radar', 'Game-by-Game Performance', 
                        'Skill Distribution', 'Win Rate Analysis'),
        specs=[
            [{"type": "polar"}, {"type": "scatter"}],
            [{"type": "bar"}, {"type": "pie"}]
        ]
    )
    
    stats = ['Hits', 'Throws', 'Blocks', 'Dodges', 'Catches']
    avg_stats = player_data[stats].mean()
    
    fig.add_trace(go.Scatterpolar(
        r=avg_stats.values,
        theta=stats,
        fill='toself',
        name='Avg Skills',
        line_color='#FF6B6B'
    ), row=1, col=1)
    
    fig.add_trace(go.Scatter(
        x=player_data['Game_ID'],
        y=player_data['Overall_Performance'],
        mode='lines+markers',
        name='Performance Trend',
        line=dict(color='#4ECDC4', width=3)
    ), row=1, col=2)
    
    fig.add_trace(go.Bar(
        x=stats,
        y=avg_stats.values,
        name='Average Stats',
        marker_color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FECA57']
    ), row=2, col=1)
    
    outcomes = player_data['Game_Outcome'].value_counts()
    fig.add_trace(go.Pie(
        labels=outcomes.index,
        values=outcomes.values,
        name="Win Rate",
        marker_colors=['#4ECDC4', '#FF6B6B']
    ), row=2, col=2)
    
    fig.update_layout(height=800, showlegend=False, 
                      title_text=f"Comprehensive Dashboard: {player_id}")
    
    return fig

def create_team_analytics(df, team_id):
    """Create detailed team analytics visualization."""
    team_data = df[df['Team'] == team_id]
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Team Performance Distribution', 'Player Roles', 
                        'Game Outcomes', 'Offensive vs. Defensive Rating'),
        specs=[[{"type": "histogram"}, {"type": "bar"}],
               [{"type": "pie"}, {"type": "scatter"}]]
    )
    
    fig.add_trace(go.Histogram(
        x=team_data['Overall_Performance'],
        nbinsx=15,
        name='Performance Distribution',
        marker_color='#4ECDC4'
    ), row=1, col=1)
    
    role_counts = team_data['Player_Role'].value_counts()
    fig.add_trace(go.Bar(
        x=role_counts.index,
        y=role_counts.values,
        name='Player Roles',
        marker_color='#FF6B6B'
    ), row=1, col=2)
    
    outcomes = team_data['Game_Outcome'].value_counts()
    fig.add_trace(go.Pie(
        labels=outcomes.index,
        values=outcomes.values,
        name="Outcomes"
    ), row=2, col=1)
    
    fig.add_trace(go.Scatter(
        x=team_data['Offensive_Rating'],
        y=team_data['Defensive_Rating'],
        mode='markers',
        text=team_data['Player_ID'],
        name='Off vs Def Rating',
        marker=dict(size=10, color=team_data['Overall_Performance'], colorscale='Viridis', showscale=True)
    ), row=2, col=2)
    
    fig.update_layout(height=800, title_text=f"Team Analytics: {team_id}", showlegend=False)
    return fig


def create_league_overview(df):
    """Create comprehensive league overview."""
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Top Performers by Avg Score', 'Team Skill Comparison', 
                        'League Role Distribution', 'Performance vs Consistency'),
        specs=[[{"type": "bar"}, {"type": "polar"}],
               [{"type": "pie"}, {"type": "scatter"}]]
    )

    top_players = df.groupby('Player_ID')['Overall_Performance'].mean().nlargest(10)
    fig.add_trace(go.Bar(
        x=top_players.index,
        y=top_players.values,
        name='Top Performers',
        marker_color='#FF6B6B'
    ), row=1, col=1)

    teams = df['Team'].unique()[:5]
    colors = px.colors.qualitative.Plotly

    for i, team in enumerate(teams):
        team_stats = df[df['Team'] == team][['Hits', 'Throws', 'Blocks', 'Dodges', 'Catches']].mean()
        fig.add_trace(go.Scatterpolar(
            r=team_stats.values,
            theta=['Hits', 'Throws', 'Blocks', 'Dodges', 'Catches'],
            fill='toself',
            name=team,
            line_color=colors[i]
        ), row=1, col=2)

    role_counts = df['Player_Role'].value_counts()
    fig.add_trace(go.Pie(
        labels=role_counts.index,
        values=role_counts.values,
        name="Roles"
    ), row=2, col=1)

    player_summary = df.groupby('Player_ID').agg({
        'Overall_Performance': 'mean',
        'Win_Rate': 'first',
        'Consistency_Score': 'first'
    }).reset_index().dropna()

    fig.add_trace(go.Scatter(
        x=player_summary['Overall_Performance'],
        y=player_summary['Consistency_Score'],
        mode='markers',
        text=player_summary['Player_ID'],
        marker=dict(
            size=player_summary['Win_Rate'] * 20 + 5,
            color=player_summary['Win_Rate'],
            colorscale='Viridis',
            showscale=True,
            colorbar_title='Win Rate'
        ),
        name='Performance vs Consistency'
    ), row=2, col=2)

    fig.update_layout(
        height=800,
        title_text="League Overview Dashboard",
        polar=dict(
            radialaxis=dict(visible=True, range=[0, df[['Hits', 'Throws', 'Blocks', 'Dodges', 'Catches']].max().max()])
        )
    )

    return fig

# --- SPECIALIZATION AND COACHING FUNCTIONS ---
def create_specialization_analysis(df):
    """Creates visualizations to analyze player specialization."""
    st.header("Player Specialization Analysis")
    st.write("This section identifies players who are specialists in key skills by comparing their performance against the league average.")

    spec_stats = ['Hits', 'Throws', 'Dodges', 'Catches', 'Hit_Accuracy', 'Defensive_Efficiency']
    player_avg_stats = df.groupby('Player_ID')[spec_stats].mean()
    
    league_avg = player_avg_stats.mean()
    
    specialization = player_avg_stats / (league_avg + 1e-6)
    
    top_specialized_players = specialization.std(axis=1).nlargest(20).index
    specialization_subset = specialization.loc[top_specialized_players]

    fig = px.imshow(specialization_subset,
                    text_auto=".2f",
                    aspect="auto",
                    color_continuous_scale='Viridis',
                    labels=dict(x="Statistic", y="Player", color="Specialization Score (x League Avg)"),
                    title="Player Specialization Heatmap (vs. League Average)")
    fig.update_xaxes(side="top")
    
    st.plotly_chart(fig, use_container_width=True)
    st.info("💡 This heatmap shows how each player's stats compare to the league average. A score of 2.0 means the player is twice as good as the average player in that specific skill.")

    st.subheader("Top Specialists by Key Skill")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.write("🛡️ **Top Catchers**")
        top_catchers = specialization.sort_values('Catches', ascending=False).head(5)
        st.dataframe(top_catchers[['Catches']].style.format("{:.2f}x Avg").background_gradient(cmap='Greens'))

    with col2:
        st.write("🏃 **Top Evaders (Dodges)**")
        top_dodgers = specialization.sort_values('Dodges', ascending=False).head(5)
        st.dataframe(top_dodgers[['Dodges']].style.format("{:.2f}x Avg").background_gradient(cmap='Blues'))

    with col3:
        st.write("🎯 **Top Sharpshooters (Hit Accuracy)**")
        top_accurate = specialization.sort_values('Hit_Accuracy', ascending=False).head(5)
        st.dataframe(top_accurate[['Hit_Accuracy']].style.format("{:.2f}x Avg").background_gradient(cmap='Reds'))

def create_coaching_comparison_chart(player_stats, role_stats, league_stats, player_id):
    """Creates a bar chart comparing a player's stats to role and league averages."""
    stats_to_compare = ['Hits', 'Throws', 'Catches', 'Dodges', 'Hit_Accuracy', 'Defensive_Efficiency']
    
    player_values = [player_stats.get(stat, 0) for stat in stats_to_compare]
    role_values = [role_stats.get(stat, 0) for stat in stats_to_compare]
    league_values = [league_stats.get(stat, 0) for stat in stats_to_compare]

    fig = go.Figure(data=[
        go.Bar(name=f'{player_id} (You)', x=stats_to_compare, y=player_values, marker_color='#FF6B6B'),
        go.Bar(name='Role Average', x=stats_to_compare, y=role_values, marker_color='#4ECDC4'),
        go.Bar(name='League Average', x=stats_to_compare, y=league_values, marker_color='#45B7D1')
    ])
    fig.update_layout(barmode='group', title_text='Performance Comparison', xaxis_title="Statistic", yaxis_title="Average Value")
    return fig

def generate_player_coaching_report(df, player_id):
    """Generates a coaching report for a single player."""
    player_data = df[df['Player_ID'] == player_id]
    if player_data.empty:
        return ["No data for this player."], None

    player_avg_stats = player_data.mean(numeric_only=True)
    player_role = df[df['Player_ID'] == player_id]['Player_Role'].iloc[0]
    
    role_avg_stats = df[df['Player_Role'] == player_role].mean(numeric_only=True)
    league_avg_stats = df.mean(numeric_only=True)

    stats_to_compare = ['Hits', 'Throws', 'Catches', 'Dodges', 'Hit_Accuracy', 'Defensive_Efficiency']
    
    # Role-specific weakness
    role_weaknesses = {}
    for stat in stats_to_compare:
        diff = (player_avg_stats.get(stat, 0) - role_avg_stats.get(stat, 0)) / (role_avg_stats.get(stat, 0) + 1e-6)
        if diff < -0.1:
            role_weaknesses[stat] = diff

    # Overall weakness
    overall_weaknesses = {}
    for stat in stats_to_compare:
        diff = (player_avg_stats.get(stat, 0) - league_avg_stats.get(stat, 0)) / (league_avg_stats.get(stat, 0) + 1e-6)
        if diff < -0.1:
            overall_weaknesses[stat] = diff

    report = [f"### Coaching Focus for {player_id} ({player_role})"]
    
    advice_map = {
        'Hit_Accuracy': "🎯 **Suggestion**: Focus on throwing drills. Practice aiming for smaller targets to improve precision under pressure.",
        'Defensive_Efficiency': "🙌 **Suggestion**: Improve decision-making when targeted. Practice drills that force a quick choice between a safe dodge and a high-reward catch.",
        'Catches': "🛡️ **Suggestion**: Improve positioning and anticipation. During games, try to predict where the opponent will throw.",
        'Dodges': "🏃 **Suggestion**: Enhance agility and footwork. Ladder drills and cone drills can improve quickness.",
        'Hits': "💥 **Suggestion**: Be more aggressive offensively. Look for opportunities to make impactful throws.",
        'Throws': "💪 **Suggestion**: Increase throwing volume without sacrificing too much accuracy."
    }

    if not role_weaknesses and not overall_weaknesses:
        report.append("✅ **Well-Rounded Performer**: This player is performing at or above average in all key areas. Great work!")
        return report, create_coaching_comparison_chart(player_avg_stats, role_avg_stats, league_avg_stats, player_id)

    if role_weaknesses:
        role_weakness_stat = min(role_weaknesses, key=role_weaknesses.get)
        report.append(f"**Role-Specific Weakness**: **{role_weakness_stat}**. Compared to other players in the '{player_role}' role, this is the biggest area for improvement.")
        report.append(advice_map.get(role_weakness_stat))

    if overall_weaknesses:
        overall_weakness_stat = min(overall_weaknesses, key=overall_weaknesses.get)
        report.append(f"**Overall Weakness**: **{overall_weakness_stat}**. Compared to the entire league, this is a key area to focus on for fundamental improvement.")
        if overall_weakness_stat != role_weaknesses.get(role_weakness_stat):
             report.append(advice_map.get(overall_weakness_stat))

    return report, create_coaching_comparison_chart(player_avg_stats, role_avg_stats, league_avg_stats, player_id)


def generate_team_coaching_report(df, team_id):
    """Generates a coaching report for a team."""
    team_data = df[df['Team'] == team_id]
    if team_data.empty:
        return ["No data for this team."]

    team_avg_stats = team_data.mean(numeric_only=True)
    league_avg_stats = df[df['Team'] != team_id].mean(numeric_only=True)

    stats_to_compare = ['Hits', 'Throws', 'Catches', 'Dodges', 'Hit_Accuracy', 'Defensive_Efficiency', 'Overall_Performance']
    weaknesses = {}
    for stat in stats_to_compare:
        diff = (team_avg_stats.get(stat, 0) - league_avg_stats.get(stat, 0)) / (league_avg_stats.get(stat, 0) + 1e-6)
        weaknesses[stat] = diff

    biggest_weakness_stat = min(weaknesses, key=weaknesses.get)

    report = [f"### Coaching Focus for {team_id}", f"**Biggest Team Weakness**: The team's **{biggest_weakness_stat}** is the furthest below the league average."]

    advice_map = {
        'Hit_Accuracy': "🎯 **Team Focus**: Dedicate a session to throwing accuracy. Set up target practice zones.",
        'Defensive_Efficiency': "🙌 **Team Focus**: Run drills that simulate 2-on-1 situations to force smart defensive decisions.",
        'Catches': "🛡️ **Team Focus**: Emphasize the value of catching to regain players and shift momentum.",
        'Dodges': "🏃 **Team Focus**: A full-team agility session with ladders and reaction games could be beneficial.",
        'Overall_Performance': "📈 **Team Focus**: Go back to basics. Focus on fundamental drills covering all areas.",
        'Defensive_Rating': "🧱 **Team Focus**: Focus on coordinated defense, communication, and protecting teammates.",
        'Offensive_Rating': "💥 **Team Focus**: Practice aggressive, coordinated attacks and identifying weaker opponents."
    }
    report.append(advice_map.get(biggest_weakness_stat))

    role_counts = team_data['Player_Role'].value_counts()
    has_catcher = any('Catcher' in str(role) for role in role_counts.index)
    if not has_catcher:
         report.append("\n**Strategic Gap**: The team lacks a dedicated 'Catcher' type player. Consider training a player for this role.")

    return report

def create_coaching_corner(df):
    """UI for the coaching corner."""
    st.header("🧑‍🏫 Coaching Corner")
    st.write("Get AI-powered, actionable advice to improve player and team performance.")

    coach_mode = st.radio("Select Coaching Mode", ["Player Coaching", "Team Coaching"], horizontal=True, key="coach_mode")

    if coach_mode == "Player Coaching":
        player_list = sorted(df['Player_ID'].unique())
        selected_player = st.selectbox("Select a Player to Coach", player_list, key="coach_player_select")
        if selected_player:
            report, fig = generate_player_coaching_report(df, selected_player)
            
            st.markdown('<div class="insight-box">', unsafe_allow_html=True)
            for line in report:
                st.markdown(line)
            st.markdown('</div>', unsafe_allow_html=True)
            
            if fig:
                st.plotly_chart(fig, use_container_width=True)

    elif coach_mode == "Team Coaching":
        team_list = sorted(df['Team'].unique())
        selected_team = st.selectbox("Select a Team to Coach", team_list, key="coach_team_select")
        if selected_team:
            report = generate_team_coaching_report(df, selected_team)
            st.markdown('<div class="insight-box">', unsafe_allow_html=True)
            for line in report:
                st.markdown(line)
            st.markdown('</div>', unsafe_allow_html=True)


# --- AI Insights Generation ---
def generate_insights(df, models):
    """Generate AI-powered insights from the data."""
    insights = []
    
    top_performer = df.groupby('Player_ID')['Overall_Performance'].mean().idxmax()
    top_score = df.groupby('Player_ID')['Overall_Performance'].mean().max()
    insights.append(f"🏆 **Top Performer**: {top_performer} with an average performance score of {top_score:.2f}")
    
    team_performance = df.groupby('Team').agg({
        'Overall_Performance': 'mean',
        'Game_Outcome': lambda x: (x == 'Win').mean()
    })
    best_team = team_performance['Overall_Performance'].idxmax()
    insights.append(f"🥇 **Strongest Team**: {best_team} with the highest average performance.")
    
    role_performance = df.groupby('Player_Role')['Overall_Performance'].mean().sort_values(ascending=False)
    best_role = role_performance.index[0]
    insights.append(f"⚡ **Most Effective Role**: {best_role}s show the highest average performance.")
    
    spec_stats = ['Catches', 'Dodges', 'Hit_Accuracy']
    player_avg_stats = df.groupby('Player_ID')[spec_stats].mean()
    league_avg = player_avg_stats.mean()
    specialization = player_avg_stats / (league_avg + 1e-6)
    
    top_catcher = specialization['Catches'].idxmax()
    top_catcher_score = specialization['Catches'].max()
    insights.append(f"🛡️ **Top Specialist**: {top_catcher} is the league's top 'Catcher', with {top_catcher_score:.1f}x the average number of catches, making them a key defensive asset.")
    
    if 'outcome_model' in models:
        model, le, accuracy, _ = models['outcome_model']
        insights.append(f"🤖 **AI Model Accuracy**: The game outcome prediction model achieves {accuracy:.1%} accuracy on test data.")
    
    return insights

# --- Main Application ---
def main():
    st.markdown("""
    <div class="main-header">
        <h1>🤾 Advanced Dodgeball Analytics Dashboard</h1>
        <p>Professional-grade performance analysis with AI-powered insights</p>
    </div>
    """, unsafe_allow_html=True)
    
    df = load_and_enhance_data('dodgeball_data2.csv')
    
    if df is None:
        st.stop()
    
    with st.spinner("Analyzing data and training AI models..."):
        df_enhanced, models = train_advanced_models(df)
    
    st.sidebar.header("🎛️ Dashboard Controls")
    
    analysis_mode = st.sidebar.selectbox(
        "Choose Analysis Mode",
        ["🏠 League Overview", "👤 Player Analysis", "🏆 Team Analysis", 
         "🤖 AI Insights", "📊 Advanced Analytics", "🧑‍🏫 Coaching Corner"]
    )
    
    if analysis_mode == "🏠 League Overview":
        st.header("League Overview")
        
        # --- REMOVED CLUTCH RATING METRIC ---
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown('<div class="metric-container">', unsafe_allow_html=True)
            st.metric("Total Players", len(df_enhanced['Player_ID'].unique()))
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="metric-container">', unsafe_allow_html=True)
            st.metric("Total Teams", len(df_enhanced['Team'].unique()))
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col3:
            st.markdown('<div class="metric-container">', unsafe_allow_html=True)
            st.metric("Games Analyzed", len(df_enhanced['Game_ID'].unique()))
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col4:
            st.markdown('<div class="metric-container">', unsafe_allow_html=True)
            avg_performance = df_enhanced['Overall_Performance'].mean()
            st.metric("Avg Performance", f"{avg_performance:.2f}")
            st.markdown('</div>', unsafe_allow_html=True)
        
        st.plotly_chart(create_league_overview(df_enhanced), use_container_width=True)
        
        st.subheader("🏆 Leaderboards")
        
        # --- REMOVED CLUTCH RATING TAB ---
        tab1, tab2, tab3 = st.tabs(["Overall Performance", "Hit Accuracy", "Win Rate"])
        
        with tab1:
            leaderboard = df_enhanced.groupby('Player_ID').agg({
                'Overall_Performance': 'mean',
                'Player_Role': 'first',
                'Team': 'first'
            }).round(2).sort_values('Overall_Performance', ascending=False).head(10)
            st.dataframe(leaderboard, use_container_width=True)
        
        with tab2:
            accuracy_board = df_enhanced.groupby('Player_ID').agg({
                'Hit_Accuracy': 'mean',
                'Player_Role': 'first',
                'Team': 'first'
            }).round(3).sort_values('Hit_Accuracy', ascending=False).head(10)
            st.dataframe(accuracy_board, use_container_width=True)
        
        with tab3:
            win_rate_board = df_enhanced.groupby('Player_ID').agg({
                'Win_Rate': 'first',
                'Player_Role': 'first',
                'Team': 'first'
            }).round(3).sort_values('Win_Rate', ascending=False).head(10)
            st.dataframe(win_rate_board, use_container_width=True)
            
    elif analysis_mode == "👤 Player Analysis":
        st.header("Individual Player Analysis")
        
        player_list = sorted(df_enhanced['Player_ID'].unique())
        selected_player = st.selectbox("Select Player", player_list)
        
        if selected_player:
            player_data = df_enhanced[df_enhanced['Player_ID'] == selected_player].iloc[0]
            
            # --- REMOVED CLUTCH RATING METRIC ---
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.markdown('<div class="metric-container">', unsafe_allow_html=True)
                st.metric("Player Role", player_data['Player_Role'])
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col2:
                st.markdown('<div class="metric-container">', unsafe_allow_html=True)
                st.metric("Win Rate", f"{player_data['Win_Rate']:.1%}")
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col3:
                st.markdown('<div class="metric-container">', unsafe_allow_html=True)
                st.metric("Avg Hit Accuracy", f"{player_data['Avg_Hit_Accuracy']:.1%}")
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col4:
                st.markdown('<div class="metric-container">', unsafe_allow_html=True)
                st.metric("Avg Performance", f"{player_data['Avg_Performance']:.2f}")
                st.markdown('</div>', unsafe_allow_html=True)
                
            st.plotly_chart(create_player_dashboard(df_enhanced, selected_player), 
                            use_container_width=True)
            
            st.subheader("Player Comparison")
            compare_check = st.checkbox("Enable Player Comparison")
            if compare_check:
                comparison_player = st.selectbox("Compare with:", 
                                                 [p for p in player_list if p != selected_player])
                
                if comparison_player:
                    comp_col1, comp_col2 = st.columns(2)
                    
                    with comp_col1:
                        st.plotly_chart(create_player_dashboard(df_enhanced, selected_player), 
                                        use_container_width=True, key="compare_player_1")
                    
                    with comp_col2:
                        st.plotly_chart(create_player_dashboard(df_enhanced, comparison_player), 
                                        use_container_width=True, key="compare_player_2")
    
    elif analysis_mode == "🏆 Team Analysis":
        st.header("Team Analysis")
        
        team_list = sorted(df_enhanced['Team'].unique())
        selected_team = st.selectbox("Select Team", team_list)
        
        if selected_team:
            team_data = df_enhanced[df_enhanced['Team'] == selected_team]
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.markdown('<div class="metric-container">', unsafe_allow_html=True)
                st.metric("Team Size", len(team_data['Player_ID'].unique()))
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col2:
                st.markdown('<div class="metric-container">', unsafe_allow_html=True)
                win_rate = (team_data['Game_Outcome'] == 'Win').mean()
                st.metric("Team Win Rate", f"{win_rate:.1%}")
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col3:
                st.markdown('<div class="metric-container">', unsafe_allow_html=True)
                avg_perf = team_data['Overall_Performance'].mean()
                st.metric("Avg Performance", f"{avg_perf:.2f}")
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col4:
                st.markdown('<div class="metric-container">', unsafe_allow_html=True)
                consistency = team_data['Consistency_Score'].mean()
                st.metric("Team Consistency", f"{consistency:.2f}")
                st.markdown('</div>', unsafe_allow_html=True)
            
            st.plotly_chart(create_team_analytics(df_enhanced, selected_team), 
                            use_container_width=True)
            
            st.subheader("Team Roster")
            roster = team_data.groupby('Player_ID').agg({
                'Player_Role': 'first',
                'Overall_Performance': 'mean',
                'Win_Rate': 'first',
                'Hit_Accuracy': 'mean'
            }).round(3).sort_values('Overall_Performance', ascending=False)
            st.dataframe(roster, use_container_width=True)
    
    elif analysis_mode == "🤖 AI Insights":
        st.header("AI-Powered Insights")
        
        insights = generate_insights(df_enhanced, models)
        
        for insight in insights:
            st.markdown(f'<div class="insight-box">{insight}</div>', unsafe_allow_html=True)
        
        st.subheader("Model Performance Metrics")
        
        if 'outcome_model' in models:
            model, le, accuracy, features = models['outcome_model']
            
            col1, col2 = st.columns([1,1])
            
            with col1:
                st.metric("Game Outcome Prediction Accuracy", f"{accuracy:.1%}")
                
                feature_importance = pd.DataFrame({
                    'feature': features,
                    'importance': model.feature_importances_
                }).sort_values('importance', ascending=False)
                
                fig = px.bar(feature_importance, x='importance', y='feature', 
                             title='Feature Importance for Game Outcome',
                             color='importance', color_continuous_scale='viridis')
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.subheader("Make a Prediction")
                st.write("Enter team-average stats to predict game outcome:")
                
                pred_hits = st.number_input("Hits", min_value=0, max_value=20, value=5)
                pred_throws = st.number_input("Throws", min_value=0, max_value=30, value=12)
                pred_dodges = st.number_input("Dodges", min_value=0, max_value=20, value=8)
                pred_catches = st.number_input("Catches", min_value=0, max_value=10, value=2)
                
                if st.button("Predict Outcome"):
                    hit_accuracy = pred_hits / pred_throws if pred_throws > 0 else 0
                    defensive_efficiency = (pred_catches + pred_dodges) / (pred_catches + pred_dodges + (pred_hits * 0.5)) if (pred_catches + pred_dodges + (pred_hits * 0.5)) > 0 else 0
                    offensive_rating = (pred_hits * 2 + pred_throws * 0.5) / (pred_throws + 1)
                    defensive_rating = (pred_dodges + pred_catches * 2) / 3
                    overall_performance = (offensive_rating * 0.4 + defensive_rating * 0.4 + 
                                           hit_accuracy * 0.1 + defensive_efficiency * 0.1)
                    
                    prediction_data = [[pred_hits, pred_throws, pred_dodges, 
                                        pred_catches, overall_performance, offensive_rating, 
                                        defensive_rating]]
                    
                    prediction = model.predict(prediction_data)[0]
                    probability = model.predict_proba(prediction_data)[0].max()
                    
                    outcome = le.inverse_transform([prediction])[0]
                    
                    if outcome == 'Win':
                        st.success(f"Predicted Outcome: **{outcome}** (Confidence: {probability:.1%})")
                    else:
                        st.error(f"Predicted Outcome: **{outcome}** (Confidence: {probability:.1%})")
    
    elif analysis_mode == "📊 Advanced Analytics":
        st.header("Advanced Analytics")
        
        tab1, tab2, tab3, tab4 = st.tabs(["Correlation Analysis", "Performance Distributions", "Key Metric Analysis", "Player Specialization"])
        
        with tab1:
            st.subheader("Performance Metrics Correlation Matrix")
            numeric_cols = ['Hits', 'Throws', 'Dodges', 'Catches', 
                            'Hit_Accuracy', 'Defensive_Efficiency', 'Overall_Performance']
            corr_matrix = df_enhanced[numeric_cols].corr()
            
            fig = px.imshow(corr_matrix, 
                            text_auto=True,
                            title="How different stats relate to each other",
                            color_continuous_scale='RdBu_r',
                            aspect="auto")
            st.plotly_chart(fig, use_container_width=True)
        
        with tab2:
            st.subheader("Performance Distribution by Role")
            
            fig = px.box(df_enhanced, x='Player_Role', y='Overall_Performance',
                         title='Performance Distribution by Player Role',
                         color='Player_Role')
            st.plotly_chart(fig, use_container_width=True)
            
            st.subheader("Win Rate by Team")
            team_winrates = df_enhanced.groupby('Team').agg({
                'Game_Outcome': lambda x: (x == 'Win').mean()
            }).reset_index().sort_values('Game_Outcome', ascending=False)
            team_winrates.columns = ['Team', 'Win_Rate']
            
            fig = px.bar(team_winrates, x='Team', y='Win_Rate',
                         title='Team Win Rates',
                         color='Win_Rate', color_continuous_scale='viridis',
                         text_auto='.1%')
            st.plotly_chart(fig, use_container_width=True)
        
        with tab3:
            st.subheader("Statistical Analysis")
            
            role_stats = df_enhanced.groupby('Player_Role')['Overall_Performance'].describe()
            st.write("Performance Statistics by Role:")
            st.dataframe(role_stats.round(3))
            
            st.subheader("Strongest Performance Correlations")
            
            numeric_cols = ['Hits', 'Throws', 'Dodges', 'Catches', 
                            'Hit_Accuracy', 'Defensive_Efficiency', 'Overall_Performance']
            performance_corr = df_enhanced[numeric_cols].corr()['Overall_Performance'].abs().sort_values(ascending=False)[1:6]
            
            corr_df = pd.DataFrame({
                'Metric': performance_corr.index,
                'Correlation': performance_corr.values
            })
            
            fig = px.bar(corr_df, x='Correlation', y='Metric',
                         title='Top 5 Metrics Correlated with Overall Performance',
                         color='Correlation', color_continuous_scale='plasma',
                         text_auto='.2f')
            st.plotly_chart(fig, use_container_width=True)

        with tab4:
            create_specialization_analysis(df_enhanced)

    elif analysis_mode == "🧑‍🏫 Coaching Corner":
        create_coaching_corner(df_enhanced)


if __name__ == "__main__":
    main()
