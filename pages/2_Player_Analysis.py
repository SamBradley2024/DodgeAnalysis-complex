import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
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
    # This is the per-game data for the player
    player_all_games = df[df['Player_ID'] == selected_player].copy()
    # This is the career average data for the player
    career_summary = player_all_games.mean(numeric_only=True)

    # --- Career Summary Metrics ---
    st.subheader(f"Career Averages for {selected_player}")
    col1, col2, col3, col4 = st.columns(4)
    with col1: utils.styled_metric("Avg Performance", f"{career_summary.get('Overall_Performance', 0):.2f}")
    with col2: utils.styled_metric("Avg K/D Ratio", f"{career_summary.get('K_D_Ratio', 0):.2f}")
    with col3: utils.styled_metric("Avg Hit Accuracy", f"{career_summary.get('Hit_Accuracy', 0):.1%}")
    with col4: utils.styled_metric("Games Played", f"{player_all_games['Game_ID'].nunique()}")
    st.markdown("---")

    tab1, tab2, tab3 = st.tabs(["Situational Performance", "Elimination Profile", "Game-by-Game Trends"])

    with tab1:
        st.subheader("Situational Performance (Career Averages)")
        # --- RESTORED: Offensive Chart ---
        offensive_df = pd.DataFrame({
            'Situation': ['Singles', 'Multi-ball', 'Counters'],
            'Hits': [career_summary.get('Hits_Singles', 0), career_summary.get('Hits_Multi', 0), career_summary.get('Hits_Counters', 0)],
            'Throws': [career_summary.get('Throws_Singles', 0), career_summary.get('Throws_Multi', 0), career_summary.get('Throws_Counters', 0)]
        })
        offensive_df['Hit_Accuracy'] = (offensive_df['Hits'] / offensive_df['Throws'].replace(0, 1)) * 100
        fig_offense = make_subplots(specs=[[{"secondary_y": True}]])
        fig_offense.add_trace(go.Bar(x=offensive_df['Situation'], y=offensive_df['Hits'], name='Avg Hits', marker_color='#1f77b4'), secondary_y=False)
        fig_offense.add_trace(go.Bar(x=offensive_df['Situation'], y=offensive_df['Throws'], name='Avg Throws', marker_color='#aec7e8'), secondary_y=False)
        fig_offense.add_trace(go.Scatter(x=offensive_df['Situation'], y=offensive_df['Hit_Accuracy'], name='Hit Accuracy (%)', mode='lines+markers', line=dict(color='red', width=4)), secondary_y=True)
        fig_offense.update_layout(title_text='<b>Offensive Performance by Situation</b>', barmode='group')
        fig_offense.update_yaxes(title_text="Average Count", secondary_y=False)
        fig_offense.update_yaxes(title_text="Hit Accuracy (%)", secondary_y=True, range=[0, 101])
        st.plotly_chart(fig_offense, use_container_width=True)

        # --- RESTORED: Defensive Chart ---
        defensive_df = pd.DataFrame({
            'Situation': ['Singles', 'Multi-ball', 'Counters'],
            'Dodges': [career_summary.get('Dodges_Singles', 0), career_summary.get('Dodges_Multi', 0), career_summary.get('Dodges_Counters', 0)],
            'Blocks': [career_summary.get('Blocks_Singles', 0), career_summary.get('Blocks_Multi', 0), career_summary.get('Blocks_Counters', 0)],
            'Survivability': [career_summary.get('Survivability_Singles', 0), career_summary.get('Survivability_Multi', 0), career_summary.get('Survivability_Counters', 0)]
        })
        fig_defense = make_subplots(specs=[[{"secondary_y": True}]])
        fig_defense.add_trace(go.Bar(x=defensive_df['Situation'], y=defensive_df['Dodges'], name='Avg Dodges', marker_color='#2ca02c'), secondary_y=False)
        fig_defense.add_trace(go.Bar(x=defensive_df['Situation'], y=defensive_df['Blocks'], name='Avg Blocks', marker_color='#98df8a'), secondary_y=False)
        fig_defense.add_trace(go.Scatter(x=defensive_df['Situation'], y=defensive_df['Survivability'], name='Survivability %', mode='lines+markers', line=dict(color='purple', width=4)), secondary_y=True)
        fig_defense.update_layout(title_text='<b>Defensive Performance by Situation</b>', barmode='group')
        fig_defense.update_yaxes(title_text="Average Count", secondary_y=False)
        fig_defense.update_yaxes(title_text="Survivability (%)", secondary_y=True, range=[0, 101])
        st.plotly_chart(fig_defense, use_container_width=True)

    with tab2:
        st.subheader("Elimination Profile (Career Totals)")
        # --- NEW: Caught Out vs. Hit Out Analysis ---
        hit_out_total = player_all_games['Hit_Out'].sum()
        caught_out_total = player_all_games['Caught_Out'].sum()
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Times Eliminated by Hit", f"{hit_out_total:.0f}")
        with col2:
            st.metric("Total Times Player's Throw Was Caught", f"{caught_out_total:.0f}")
        
        if caught_out_total > 0 or hit_out_total > 0:
            elim_pie_df = pd.DataFrame({
                'Reason': ['Eliminated by Hit', 'Throw Caught by Opponent'],
                'Count': [hit_out_total, caught_out_total]
            })
            fig_elim_pie = px.pie(elim_pie_df, values='Count', names='Reason', title='Primary Reasons for Elimination')
            st.plotly_chart(fig_elim_pie, use_container_width=True)
        else:
            st.info("This player was not eliminated in the loaded games.")

    with tab3:
        st.subheader("Game-by-Game Performance Trends")
        # --- FIXED: Trend chart now correctly plots per-game stats ---
        trend_metrics = sorted([col for col in df.columns if pd.api.types.is_numeric_dtype(df[col]) and col not in ['Role_Cluster']])
        default_index = trend_metrics.index('Hit_Accuracy') if 'Hit_Accuracy' in trend_metrics else 0
        selected_trend_metric = st.selectbox("Select metric to track:", trend_metrics, index=default_index)
        
        if len(player_all_games) > 1:
            fig_trend = px.line(
                player_all_games.sort_values('Game_ID'),
                x='Game_ID',
                y=selected_trend_metric,
                markers=True,
                title=f"{selected_trend_metric.replace('_', ' ')} Trend Across Games"
            )
            fig_trend.update_layout(xaxis_title="Game", yaxis_title=selected_trend_metric.replace('_', ' '))
            st.plotly_chart(fig_trend, use_container_width=True)
        else:
            st.info("Load more than one game to see performance trends over time.")

