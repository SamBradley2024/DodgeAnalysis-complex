import streamlit as st
import utils

# --- Page Configuration and State Check ---
st.set_page_config(page_title="League Overview", page_icon="🏠", layout="wide")
st.markdown(utils.load_css(), unsafe_allow_html=True)

if 'data_loaded' not in st.session_state or not st.session_state.data_loaded:
    st.warning("Please select and load a data source from the 🏠 Home page first.")
    st.stop()

df = st.session_state.df_enhanced
models = st.session_state.models

# --- Page Content ---
st.header("League Overview")
st.info(f"Analyzing data from: **{st.session_state.source_name}**")
st.markdown("---")

# --- Key Metrics ---
col1, col2, col3, col4 = st.columns(4)
with col1:
    utils.styled_metric("Total Players", df['Player_ID'].nunique())
with col2:
    utils.styled_metric("Total Teams", df['Team'].nunique())
with col3:
    utils.styled_metric("Games Analyzed", df['Game_ID'].nunique())
with col4:
    avg_performance = df['Overall_Performance'].mean()
    utils.styled_metric("Avg Performance", f"{avg_performance:.2f}")

# --- CORRECTED: Player Summary Aggregation ---
# This block is updated to calculate the averages from the per-game 'Overall_Performance' column.
player_summary = df.groupby('Player_ID').agg(
    Team=('Team', 'first'),
    Player_Role=('Player_Role', 'first'),
    Avg_Performance=('Overall_Performance', 'mean'),
    Avg_KD_Ratio=('K/D_Ratio', 'mean'),
    Avg_Hit_Accuracy=('Hit_Accuracy', 'mean'),
    Avg_Throws=('Throws', 'mean'),
    Avg_Dodges=('Dodges', 'mean'),
    Avg_Blocks=('Blocks', 'mean'),
    Avg_Catches=('Catches', 'mean')
).reset_index()

# --- Leaderboards ---
st.subheader("Leaderboards")
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["Performance", "K/D Ratio", "Accuracy", "Throwers", "Dodgers", "Blockers"])

with tab1:
    leaderboard = player_summary[['Player_ID', 'Team', 'Player_Role', 'Avg_Performance']].sort_values('Avg_Performance', ascending=False).head(10)
    st.dataframe(leaderboard.style.format({'Avg_Performance': '{:.2f}'}), use_container_width=True)

with tab2:
    kd_board = player_summary[['Player_ID', 'Team', 'Player_Role', 'Avg_KD_Ratio']].sort_values('Avg_KD_Ratio', ascending=False).head(10)
    st.dataframe(kd_board.style.format({'Avg_KD_Ratio': '{:.2f}'}), use_container_width=True)

with tab3:
    accuracy_board = player_summary[['Player_ID', 'Team', 'Player_Role', 'Avg_Hit_Accuracy']].sort_values('Avg_Hit_Accuracy', ascending=False).head(10)
    st.dataframe(accuracy_board.style.format({'Avg_Hit_Accuracy': '{:.1%}'}), use_container_width=True)

with tab4:
    thrower_board = player_summary[['Player_ID', 'Team', 'Player_Role', 'Avg_Throws']].sort_values('Avg_Throws', ascending=False).head(10)
    st.dataframe(thrower_board.style.format({'Avg_Throws': '{:.2f}'}), use_container_width=True)

with tab5:
    dodger_board = player_summary[['Player_ID', 'Team', 'Player_Role', 'Avg_Dodges']].sort_values('Avg_Dodges', ascending=False).head(10)
    st.dataframe(dodger_board.style.format({'Avg_Dodges': '{:.2f}'}), use_container_width=True)

with tab6:
    blocker_board = player_summary[['Player_ID', 'Team', 'Player_Role', 'Avg_Catches']].sort_values('Avg_Catches', ascending=False).head(10)
    st.dataframe(blocker_board.style.format({'Avg_Catches': '{:.2f}'}), use_container_width=True)

# --- Visualizations ---
st.markdown("---")
fig = utils.create_league_overview(df)
st.plotly_chart(fig, use_container_width=True)
