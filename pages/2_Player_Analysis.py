import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import utils

# --- Page Configuration and State Check ---
st.set_page_config(page_title="Player Analysis", page_icon="👤", layout="wide")
st.markdown(utils.load_css(), unsafe_allow_html=True)

if 'data_loaded' not in st.session_state or not st.session_state.data_loaded:
    st.warning("Please select and load a data source from the 🏠 Home page first.")
    st.stop()

df = st.session_state.df_enhanced

# --- Player Selection ---
st.header("👤 Individual Player Analysis")
player_list = sorted(df['Player_ID'].unique())
if not player_list:
    st.warning("No players found in the selected data source.")
    st.stop()
selected_player = st.selectbox("Select Player", player_list)

if selected_player:
    # --- Data Preparation ---
    player_all_games = df[df['Player_ID'] == selected_player].copy()
    games_played = player_all_games['Game_ID'].unique()

    # --- Analysis Mode Selection ---
    st.subheader("Select Analysis Mode")
    analysis_mode = st.radio(
        "Choose to view career averages or stats from a single game:",
        ["Career Averages", "Single Game Analysis"],
        horizontal=True
    )
    st.markdown("---")

    # --- Conditional Data Source ---
    # The 'stats_source' will hold either the career averages or the single-game stats
    # The 'title_suffix' will be used to dynamically update chart titles
    stats_source = None
    title_suffix = ""

    if analysis_mode == "Career Averages":
        stats_source = player_all_games.mean(numeric_only=True)
        title_suffix = "(Career Avg)"
        st.subheader(f"Career Performance for {selected_player}")
        col1, col2, col3, col4 = st.columns(4)
        with col1: utils.styled_metric("Avg Performance", f"{stats_source.get('Overall_Performance', 0):.2f}")
        with col2: utils.styled_metric("Avg K/D Ratio", f"{stats_source.get('K/D_Ratio', 0):.2f}")
        with col3: utils.styled_metric("Avg Hit Accuracy", f"{stats_source.get('Hit_Accuracy', 0):.1%}")
        with col4: utils.styled_metric("Games Played", f"{len(games_played)}")

    elif analysis_mode == "Single Game Analysis":
        if len(games_played) > 0:
            selected_game = st.selectbox("Select a game to analyze:", games_played)
            if selected_game:
                stats_source = player_all_games[player_all_games['Game_ID'] == selected_game].iloc[0]
                title_suffix = f"(Game: {selected_game})"
                st.subheader(f"Performance for {selected_player} in {selected_game}")
        else:
            st.warning("This player has no games recorded.")
            st.stop()

    # --- Main Analysis Tabs ---
    if stats_source is not None:
        tab_list = ["Situational Performance", "Elimination Profile", "Game-by-Game Trends"]
        # Disable trend tab if viewing a single game
        if analysis_mode == "Single Game Analysis":
            tab_list.pop()

        tabs = st.tabs(tab_list)

        with tabs[0]:
            st.info(f"Displaying {analysis_mode.lower()} for the charts below.")
            # --- Offensive Analysis with Stacked Chart ---
            st.subheader("Offensive Breakdown")
            offensive_df = pd.DataFrame({
                'Situation': ['Singles', 'Multi-ball', 'Counters', 'Pres'],
                'Hits': [stats_source.get(c, 0) for c in ['Hits_Singles', 'Hits_Multi', 'Hits_Counters', 'Hits_Pres']],
                'Throws': [stats_source.get(c, 0) for c in ['Throws_Singles', 'Throws_Multi', 'Throws_Counters', 'Throws_Pres']]
            })
            offensive_df['Misses'] = offensive_df['Throws'] - offensive_df['Hits']
            offensive_df['Hit_Accuracy'] = (offensive_df['Hits'] / offensive_df['Throws'].replace(0, 1))

            fig_offense = go.Figure()
            fig_offense.add_trace(go.Bar(x=offensive_df['Situation'], y=offensive_df['Hits'], name='Hits', marker_color='green'))
            fig_offense.add_trace(go.Bar(x=offensive_df['Situation'], y=offensive_df['Misses'], name='Misses', marker_color='red'))
            fig_offense.update_layout(barmode='stack', title_text=f'<b>Offensive Performance {title_suffix}</b>', yaxis_title='Total Throws')
            for i, row in offensive_df.iterrows():
                if row['Throws'] > 0:
                    fig_offense.add_annotation(x=row['Situation'], y=row['Throws'], text=f"<b>{row['Hit_Accuracy']:.1%} Acc.</b>", showarrow=False, yshift=10, font=dict(color="black"))
            st.plotly_chart(fig_offense, use_container_width=True)

            # --- Defensive Analysis with Stacked Chart ---
            st.subheader("Defensive Breakdown")
            defensive_df = pd.DataFrame({
                'Situation': ['Singles', 'Multi-ball', 'Counters', 'Pres'],
                'Dodges': [stats_source.get(c, 0) for c in ['Dodges_Singles', 'Dodges_Multi', 'Dodges_Counters', 'Dodges_Pres']],
                'Blocks': [stats_source.get(c, 0) for c in ['Blocks_Singles', 'Blocks_Multi', 'Blocks_Counters', 'Blocks_Pres']],
                'Times_Thrown_At': [stats_source.get(c, 0) for c in ['Thrown_At_Singles', 'Thrown_At_Multi', 'Thrown_At_Counters', 'Thrown_At_Pres']],
            })
            defensive_df['Hit_Out'] = defensive_df['Times_Thrown_At'] - defensive_df['Dodges'] - defensive_df['Blocks']
            defensive_df['Survivability'] = 1 - (defensive_df['Hit_Out'] / defensive_df['Times_Thrown_At'].replace(0, 1))

            fig_defense = go.Figure()
            fig_defense.add_trace(go.Bar(x=defensive_df['Situation'], y=defensive_df['Dodges'], name='Dodges', marker_color='blue'))
            fig_defense.add_trace(go.Bar(x=defensive_df['Situation'], y=defensive_df['Blocks'], name='Blocks', marker_color='lightblue'))
            fig_defense.add_trace(go.Bar(x=defensive_df['Situation'], y=defensive_df['Hit_Out'], name='Hit Out', marker_color='orange'))
            fig_defense.update_layout(barmode='stack', title_text=f'<b>Defensive Performance {title_suffix}</b>', yaxis_title='Times Thrown At')
            for i, row in defensive_df.iterrows():
                if row['Times_Thrown_At'] > 0:
                    fig_defense.add_annotation(x=row['Situation'], y=row['Times_Thrown_At'], text=f"<b>{row['Survivability']:.1%} Surv.</b>", showarrow=False, yshift=10, font=dict(color="black"))
            st.plotly_chart(fig_defense, use_container_width=True)

        with tabs[1]:
            st.subheader("Elimination & Catching Profile")
            col1, col2 = st.columns(2)
            # Use total sums for career view, or single game stats for game view
            source_for_totals = player_all_games if analysis_mode == "Career Averages" else stats_source

            with col1:
                elim_data = { 'Reason': ['Eliminated by Hit', "Player's Throw Caught"], 'Count': [source_for_totals['Hit_Out'].sum(), source_for_totals['Caught_Out'].sum()] }
                elim_df = pd.DataFrame(elim_data)
                fig_elim = px.pie(elim_df, values='Count', names='Reason', title=f'How Player is Eliminated {title_suffix}')
                st.plotly_chart(fig_elim, use_container_width=True)
            with col2:
                catches_made = source_for_totals['Catches'].sum()
                catches_attempted = source_for_totals['Catches_Attempted'].sum()
                catch_data = { 'Result': ['Successful Catches', 'Failed Attempts'], 'Count': [catches_made, max(0, catches_attempted - catches_made)] }
                catch_df = pd.DataFrame(catch_data)
                fig_catch = px.pie(catch_df, values='Count', names='Result', title=f'Player Catching Performance {title_suffix}')
                st.plotly_chart(fig_catch, use_container_width=True)

        if analysis_mode == "Career Averages":
            with tabs[2]:
                st.subheader("Game-by-Game Trend")
                trend_metrics = sorted([col for col in df.columns if pd.api.types.is_numeric_dtype(df[col]) and col not in ['Role_Cluster']])
                default_index = trend_metrics.index('Hit_Accuracy') if 'Hit_Accuracy' in trend_metrics else 0
                selected_trend_metric = st.selectbox("Select metric to track over time:", trend_metrics, index=default_index)
                
                if selected_trend_metric:
                    fig_trend = px.line(player_all_games.sort_values('Game_ID'), x='Game_ID', y=selected_trend_metric, markers=True, title=f'{selected_player} - {selected_trend_metric.replace("_", " ")} Trend')
                    if len(player_all_games) < 2:
                        fig_trend.update_traces(mode='markers', marker=dict(size=12))
                    st.plotly_chart(fig_trend, use_container_width=True)

