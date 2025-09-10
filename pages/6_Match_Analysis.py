import streamlit as st
import pandas as pd
import plotly.express as px
import utils

# --- Page Configuration and State Check ---
st.set_page_config(page_title="Game Analysis", page_icon="🎲", layout="wide")
st.markdown(utils.load_css(), unsafe_allow_html=True)

if 'data_loaded' not in st.session_state or not st.session_state.data_loaded:
    st.warning("Please select and load a data source from the 🏠 Home page first.")
    st.stop()

df = st.session_state.df_enhanced

# --- Game Selection ---
st.header("🎲 Single Game Analysis")
game_list = sorted(df['Game_ID'].unique())
if not game_list:
    st.warning("No games found in the selected data source.")
    st.stop()

selected_game = st.selectbox("Select a Game to Analyze", game_list)

if selected_game:
    # Data for the selected game
    game_df = df[df['Game_ID'] == selected_game].copy()
    
    # Career data (all games EXCEPT the selected one) for comparison
    career_df = df[df['Game_ID'] != selected_game]

    st.subheader(f"Performance in: {selected_game}")

    # --- Calculate Career Averages for Comparison ---
    player_career_avg = career_df.groupby('Player_ID')['Overall_Performance'].mean().reset_index()
    player_career_avg = player_career_avg.rename(columns={'Overall_Performance': 'Career_Avg_Performance'})
    
    # Merge career averages into the game data
    game_summary = game_df.merge(player_career_avg, on='Player_ID', how='left').fillna(0)
    
    # Calculate performance vs career average
    game_summary['Perf_vs_Avg'] = game_summary['Overall_Performance'] - game_summary['Career_Avg_Performance']
    
    # --- Display Table ---
    st.dataframe(
        game_summary[[
            'Player_ID', 'Team', 'Overall_Performance', 'Career_Avg_Performance', 'Perf_vs_Avg',
            'Hits', 'Throws', 'Catches', 'Dodges'
        ]].sort_values('Overall_Performance', ascending=False).style.format({
            'Overall_Performance': '{:.2f}',
            'Career_Avg_Performance': '{:.2f}',
            'Perf_vs_Avg': '{:+.2f}'
        }).bar(subset=['Perf_vs_Avg'], align='mid', color=['#d65f5f', '#5fba7d']),
        use_container_width=True
    )
    st.info("`Perf_vs_Avg` shows how a player's performance in this game compares to their career average. Positive is good, negative is bad.")

    # --- Visual Comparison ---
    fig = px.bar(
        game_summary.sort_values('Perf_vs_Avg', ascending=False),
        x='Player_ID',
        y='Perf_vs_Avg',
        color='Team',
        title='Player Performance vs. Career Average for this Game',
        labels={'Perf_vs_Avg': 'Performance vs. Career Average', 'Player_ID': 'Player'}
    )
    fig.add_hline(y=0)
    st.plotly_chart(fig, use_container_width=True)

