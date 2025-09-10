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
    
    st.subheader(f"Player Stats for: {selected_game}")

    # --- Analysis Mode Selection ---
    view_mode = st.radio(
        "Select View Mode:",
        ["Simple View", "In-depth View"],
        horizontal=True,
        help="Choose 'Simple' for key stats or 'In-depth' for all available data from the sheet."
    )

    # --- Dynamic Column Selection ---
    if view_mode == "Simple View":
        display_cols = [
            'Player_ID', 'Team', 'Overall_Performance',
            'Hits', 'Throws', 'Catches', 'Dodges', 'Blocks', 'Times_Eliminated'
        ]
        st.info("Showing key performance indicators for each player in this game.")
    else: # In-depth View
        all_numeric_cols = sorted([col for col in game_df.columns if pd.api.types.is_numeric_dtype(game_df[col])])
        cols_to_exclude = ['Role_Cluster'] 
        display_cols = ['Player_ID', 'Team'] + [col for col in all_numeric_cols if col not in cols_to_exclude]
        st.info("Showing all available statistics for each player in this game.")

    display_cols = [col for col in display_cols if col in game_df.columns]
    
    # --- MODIFIED: Rotated Table Display ---
    # Prepare the DataFrame for transposition
    table_df = game_df[display_cols].set_index('Player_ID')
    
    # Transpose the DataFrame so players become columns
    transposed_df = table_df.T
    
    # Display the rotated table
    st.dataframe(transposed_df, use_container_width=True)


    st.markdown("---")
    st.subheader("Performance vs. Career Average")
    st.info("This chart shows how each player's performance in *this specific game* compares to their average performance across all other games loaded in the app.")

    # --- Career data (all games EXCEPT the selected one) for comparison ---
    career_df = df[df['Game_ID'] != selected_game]

    if not career_df.empty:
        player_career_avg = career_df.groupby('Player_ID')['Overall_Performance'].mean().reset_index()
        player_career_avg = player_career_avg.rename(columns={'Overall_Performance': 'Career_Avg_Performance'})

        game_summary = game_df.merge(player_career_avg, on='Player_ID', how='left')
        game_summary['Career_Avg_Performance'] = game_summary['Career_Avg_Performance'].fillna(game_summary['Overall_Performance'])
        game_summary['Perf_vs_Avg'] = game_summary['Overall_Performance'] - game_summary['Career_Avg_Performance']

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
    else:
        st.warning("Only one game has been loaded, so a career comparison cannot be generated.")

