import streamlit as st
import pandas as pd
import utils

# --- Page Configuration and State Check ---
st.set_page_config(page_title="League Overview", page_icon="🏆", layout="wide")
st.markdown(utils.load_css(), unsafe_allow_html=True)

if 'data_loaded' not in st.session_state or not st.session_state.data_loaded:
    st.warning("Please select and load a data source from the 🏠 Home page first.")
    st.stop()

df = st.session_state.df_enhanced

# --- Data Cleaning and Type Conversion ---
# Ensure all relevant columns are numeric, handling errors by coercing to NaN and filling with 0
for col in df.columns:
    if df[col].dtype == 'object' and col not in ['Player_ID', 'Team', 'Match_ID', 'Game_ID', 'Game_Outcome', 'Player_Role']:
        if '%' in str(df[col].iloc[0]):
            df[col] = df[col].str.replace('#DIV/0!', '0', regex=False)
            df[col] = pd.to_numeric(df[col].str.replace('%', '', regex=False), errors='coerce').fillna(0) / 100.0
        else:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

# --- Page Content ---
st.header("🏆 League Overview & Leaderboards")
st.info(f"Analyzing detailed situational data from: **{st.session_state.source_name}**")
st.markdown("---")

# --- Key Metrics ---
col1, col2, col3, col4 = st.columns(4)
with col1:
    utils.styled_metric("Total Players", df['Player_ID'].nunique())
with col2:
    utils.styled_metric("Total Teams", df['Team'].nunique())
with col3:
    utils.styled_metric("Total Sets Played", f"{df['Sets_Played'].sum():,}")
with col4:
    avg_performance = df['Overall_Performance'].mean()
    utils.styled_metric("Avg Performance Score", f"{avg_performance:.2f}")

# --- Leaderboards ---
st.subheader("Situational Leaderboards")
st.markdown("Top 10 players based on specific in-game situations.")

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🎯 Singles Specialist (Hits)", 
    "🛡️ Multi-ball Survivor", 
    "🙌 Top Catcher (Attempts)",
    "💥 Counter-Attack Hits",
    "🏃‍♂️ Top Dodger (Overall)"
])

# Use the full dataframe for leaderboards, not a summary, to show raw totals
leaderboard_cols = ['Player_ID', 'Team']

with tab1:
    st.markdown("##### **🎯 Top 10 Singles Hitters**")
    st.info("Players who land the most hits in one-on-one throwing situations.")
    leaderboard = df[leaderboard_cols + ['Hits_Singles']].sort_values('Hits_Singles', ascending=False).head(10)
    st.dataframe(leaderboard, use_container_width=True)

with tab2:
    st.markdown("##### **🛡️ Top 10 Multi-ball Survivors**")
    st.info("Players with the highest survivability percentage when multiple balls are live.")
    leaderboard = df[leaderboard_cols + ['Survivability_Multi']].sort_values('Survivability_Multi', ascending=False).head(10)
    st.dataframe(leaderboard.style.format({'Survivability_Multi': '{:.1%}'}), use_container_width=True)
    
with tab3:
    st.markdown("##### **🙌 Top 10 Catchers by Attempts**")
    st.info("Players who attempt to catch the most, indicating high defensive engagement.")
    leaderboard = df[leaderboard_cols + ['Catches_Attempted']].sort_values('Catches_Attempted', ascending=False).head(10)
    st.dataframe(leaderboard, use_container_width=True)

with tab4:
    st.markdown("##### **💥 Top 10 Counter-Attack Hitters**")
    st.info("Players who are most effective at landing hits immediately after a block or catch.")
    leaderboard = df[leaderboard_cols + ['Hits_Counters']].sort_values('Hits_Counters', ascending=False).head(10)
    st.dataframe(leaderboard, use_container_width=True)
    
with tab5:
    st.markdown("##### **🏃‍♂️ Top 10 Overall Dodgers**")
    st.info("Players with the highest total number of successful dodges across all situations.")
    leaderboard = df[leaderboard_cols + ['Dodges']].sort_values('Dodges', ascending=False).head(10)
    st.dataframe(leaderboard, use_container_width=True)
