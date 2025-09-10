import streamlit as st
import pandas as pd
import plotly.express as px
import utils

# --- Page Configuration and State Check ---
st.set_page_config(page_title="League Overview", page_icon="🏆", layout="wide")
st.markdown(utils.load_css(), unsafe_allow_html=True)

if 'data_loaded' not in st.session_state or not st.session_state.data_loaded:
    st.warning("Please select and load a data source from the 🏠 Home page first.")
    st.stop()

df = st.session_state.df_enhanced

# --- Page Content ---
st.header("🏆 League-Wide Analysis")
st.info(f"Analyzing {df['Game_ID'].nunique()} games from: **{st.session_state.source_name}**")

# --- FIXED: Player Summary Aggregation ---
# This block now correctly includes all necessary columns to prevent KeyErrors.
player_summary = df.groupby('Player_ID').agg(
    Overall_Performance=('Overall_Performance', 'mean'),
    K_D_Ratio=('K/D_Ratio', 'mean'),
    Hit_Accuracy=('Hit_Accuracy', 'mean'),
    Hits=('Hits', 'mean'),
    Throws=('Throws', 'mean'),
    Catches=('Catches', 'mean'),
    Dodges=('Dodges', 'mean'),
    Offensive_Rating=('Offensive_Rating', 'mean'), # Added for bubble chart
    Defensive_Rating=('Defensive_Rating', 'mean'), # Added for bubble chart
    Team=('Team', 'first'),
    Player_Role=('Player_Role', 'first')
).reset_index()

st.markdown("---")

# --- Dynamic Leaderboard ---
st.subheader("Dynamic Leaderboard")
leaderboard_metrics = [
    'Overall_Performance', 'K_D_Ratio', 'Hit_Accuracy', 
    'Hits', 'Throws', 'Catches', 'Dodges'
]
selected_metric = st.selectbox("Select a metric for the leaderboard:", leaderboard_metrics)

if selected_metric:
    top_10 = player_summary.sort_values(selected_metric, ascending=False).head(10)
    fig = px.bar(
        top_10,
        x='Player_ID',
        y=selected_metric,
        color='Team',
        title=f"Top 10 Players by Average {selected_metric.replace('_', ' ')}",
        text_auto='.2f'
    )
    fig.update_layout(xaxis={'categoryorder':'total descending'})
    st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

# --- Offense vs. Defense Bubble Chart ---
st.subheader("Player Profile: Offense vs. Defense")
fig2 = px.scatter(
    player_summary,
    x="Offensive_Rating",
    y="Defensive_Rating",
    size="Overall_Performance",
    color="Player_Role",
    hover_name="Player_ID",
    size_max=40,
    title="League Player Profiles"
)
st.plotly_chart(fig2, use_container_width=True)

