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
# This is a robust way to ensure all data is numeric for calculations
for col in df.columns:
    if df[col].dtype == 'object' and col not in ['Player_ID', 'Team', 'Match_ID', 'Game_ID', 'Game_Outcome', 'Player_Role']:
        df[col] = pd.to_numeric(df[col].str.replace('%', '', regex=False).str.replace('#DIV/0!', '0', regex=False), errors='coerce').fillna(0)
        if '%' in str(df[col].iloc[0]):
             df[col] = df[col] / 100.0

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
st.subheader("League Leaderboards")
st.markdown("Top 10 players based on key performance metrics.")

# Calculate Hit Accuracy for leaderboards
df['Hit_Accuracy_Singles'] = df['Hits_Singles'] / df['Throws_Singles'].replace(0, 1)
df['Hit_Accuracy_Multi'] = df['Hits_Multi'] / df['Throws_Multi'].replace(0, 1)

tab_titles = [
    "🏆 Overall Performance", "🎯 Hit Accuracy (Singles)", "💥 K/D Ratio", 
    "🙌 Top Catcher", "🏃‍♂️ Top Dodger", "🛡️ Top Blocker", "⚡ Singles Hits", "🔥 Multi-ball Hits"
]
tabs = st.tabs(tab_titles)
leaderboard_cols = ['Player_ID', 'Team']

with tabs[0]:
    leaderboard = df[leaderboard_cols + ['Overall_Performance']].sort_values('Overall_Performance', ascending=False).head(10)
    st.dataframe(leaderboard.style.format({'Overall_Performance': '{:.2f}'}), use_container_width=True)

with tabs[1]:
    leaderboard = df[leaderboard_cols + ['Hit_Accuracy_Singles']].sort_values('Hit_Accuracy_Singles', ascending=False).head(10)
    st.dataframe(leaderboard.style.format({'Hit_Accuracy_Singles': '{:.1%}'}), use_container_width=True)

with tabs[2]:
    leaderboard = df[leaderboard_cols + ['K/D_Ratio']].sort_values('K/D_Ratio', ascending=False).head(10)
    st.dataframe(leaderboard.style.format({'K/D_Ratio': '{:.2f}'}), use_container_width=True)

with tabs[3]:
    leaderboard = df[leaderboard_cols + ['Catches']].sort_values('Catches', ascending=False).head(10)
    st.dataframe(leaderboard, use_container_width=True)

with tabs[4]:
    leaderboard = df[leaderboard_cols + ['Dodges']].sort_values('Dodges', ascending=False).head(10)
    st.dataframe(leaderboard, use_container_width=True)

with tabs[5]:
    leaderboard = df[leaderboard_cols + ['Blocks']].sort_values('Blocks', ascending=False).head(10)
    st.dataframe(leaderboard, use_container_width=True)

with tabs[6]:
    leaderboard = df[leaderboard_cols + ['Hits_Singles']].sort_values('Hits_Singles', ascending=False).head(10)
    st.dataframe(leaderboard, use_container_width=True)

with tabs[7]:
    leaderboard = df[leaderboard_cols + ['Hits_Multi']].sort_values('Hits_Multi', ascending=False).head(10)
    st.dataframe(leaderboard, use_container_width=True)

