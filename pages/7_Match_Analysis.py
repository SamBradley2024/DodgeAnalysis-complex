import streamlit as st
import pandas as pd
import plotly.express as px
import utils

# --- Page Configuration and State Check ---
st.set_page_config(page_title="Match Analysis", page_icon="🎲", layout="wide")
st.markdown(utils.load_css(), unsafe_allow_html=True)

# Check if data has been loaded from the Home page
if 'data_loaded' not in st.session_state or not st.session_state.data_loaded:
    st.warning("Please select and load a data source from the 🏠 Home page first.")
    st.stop()

# If data is loaded, get it from session state
df = st.session_state.df_enhanced
models = st.session_state.models

# --- Page Content ---
st.header("🎲 Match Analysis")
st.info(f"Analyzing matches from: **{st.session_state.source_name}**")

match_list = sorted(df['Match_ID'].unique())
if not match_list:
    st.warning("No matches found in the selected data source.")
    st.stop()

selected_match = st.selectbox("Select a Match to Analyze", match_list)

if selected_match:
    match_df = df[df['Match_ID'] == selected_match].copy()
    
    if 'Match_Winner' in match_df.columns:
        match_winner = match_df['Match_Winner'].iloc[0]
        
        # --- FIXED LINE ---
        # First, find unique games won by each team, then count them.
        game_scores = match_df[match_df['Game_Outcome'] == 'Win'][['Game_ID', 'Team']].drop_duplicates()['Team'].value_counts()
        
        st.subheader(f"Match Result: {match_winner} Wins")
        if not game_scores.empty:
            cols = st.columns(len(game_scores))
            for i, (team, score) in enumerate(game_scores.items()):
                with cols[i]:
                    st.metric(f"{team} Game Wins", score)
    else:
        st.subheader("Match Result")
        st.warning("Match winner could not be determined.")

    st.markdown("---")

    view_mode = st.radio(
        "Select Analysis View",
        ("Match Summary", "Individual Player Performance"),
        horizontal=True,
        label_visibility="collapsed"
    )

    if view_mode == "Match Summary":
        st.subheader("Game-by-Game Breakdown")
        games_in_match = sorted(match_df['Game_ID'].unique())
        for game_id in games_in_match:
            game_df = match_df[match_df['Game_ID'] == game_id]
            winner_series = game_df[game_df['Game_Outcome'] == 'Win']['Team']
            winner = winner_series.iloc[0] if not winner_series.empty else "Draw/Unknown"
            game_num = game_df['Game_Num_In_Match'].iloc[0]
            with st.expander(f"**Game {game_num}: {winner} won**"):
                mvp = game_df.loc[game_df['Overall_Performance'].idxmax()]
                st.write(f"**Game MVP:** {mvp['Player_ID']} (Performance: {mvp['Overall_Performance']:.2f})")
                player_stats = game_df[['Player_ID', 'Team', 'Overall_Performance', 'Hits', 'Throws', 'Catches', 'Dodges', 'Blocks', 'Hit_Out', 'Caught_Out']]
                st.dataframe(player_stats.sort_values("Overall_Performance", ascending=False), width='stretch')

    elif view_mode == "Individual Player Performance":
        st.subheader("Match Performance Leaderboard")
    
        # 1. Aggregate stats for THIS match only
        agg_dict = {
            'Overall_Performance': 'mean',
            'Hits': 'sum', 'Throws': 'sum', 'Catches': 'sum',
            'Dodges': 'sum', 'Blocks': 'sum', 'Hit_Out': 'sum',
            'Caught_Out': 'sum', 'K/D_Ratio': 'mean'
        }
        match_player_summary = match_df.groupby(['Player_ID', 'Team']).agg(agg_dict).reset_index()

        # Rename the aggregated columns to be more descriptive
        match_player_summary = match_player_summary.rename(columns={
            'Overall_Performance': 'Avg_Match_Performance',
            'Hits': 'Total_Hits', 'Throws': 'Total_Throws',
            'Catches': 'Total_Catches', 'Dodges': 'Total_Dodges',
            'Blocks': 'Total_Blocks', 'Hit_Out': 'Total_Hit_Out',
            'Caught_Out': 'Total_Caught_Out', 'K/D_Ratio': 'Avg_KD_Ratio'
        })

        # 2. Calculate Career Average Performance for each player from the main 'df'
        career_avg_perf = df.groupby('Player_ID')['Overall_Performance'].mean().reset_index()
        career_avg_perf = career_avg_perf.rename(columns={'Overall_Performance': 'Career_Avg_Performance'})

        # 3. Merge career averages into the match summary
        match_player_summary = pd.merge(match_player_summary, career_avg_perf, on='Player_ID', how='left')
    
        # 4. Now, safely calculate the comparison column
        match_player_summary['Perf. vs. Avg.'] = match_player_summary['Avg_Match_Performance'] - match_player_summary['Career_Avg_Performance']
    
        match_player_summary = match_player_summary.sort_values('Avg_Match_Performance', ascending=False)
    
        display_cols = [
            'Player_ID', 'Team', 'Avg_Match_Performance', 'Perf. vs. Avg.', 
            'Total_Hits', 'Total_Throws', 'Total_Catches', 'Total_Dodges', 
            'Total_Blocks', 'Total_Hit_Out', 'Total_Caught_Out', 'Avg_KD_Ratio'
        ]
    
        st.dataframe(match_player_summary[display_cols].style.format({
            'Avg_Match_Performance': '{:.2f}',
            'Perf. vs. Avg.': '{:+.2f}',
            'Avg_KD_Ratio': '{:.2f}'
        }).bar(subset=['Perf. vs. Avg.'], align='mid', color=['#d65f5f', '#5fba7d']),
        width='stretch')

        fig = px.bar(
            match_player_summary, x='Player_ID', y='Avg_Match_Performance', color='Team',
            title='Player Performance Scores for the Match',
            labels={'Player_ID': 'Player', 'Avg_Match_Performance': 'Average Performance in Match'}
        )
        st.plotly_chart(fig, width='stretch')