import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
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
player_list = sorted(df['Player_ID'].unique())
if not player_list:
    st.warning("No players found in the selected data source.")
    st.stop()
    
st.header("👤 Individual Player Analysis")
selected_player = st.selectbox("Select Player", player_list)

if selected_player:
    # Filter data for the selected player across all games
    player_all_games = df[df['Player_ID'] == selected_player].copy()
    
    # --- Career Summary Metrics ---
    st.subheader(f"Career Averages for {selected_player}")
    career_summary = player_all_games.mean(numeric_only=True)
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        utils.styled_metric("Avg Performance", f"{career_summary.get('Overall_Performance', 0):.2f}")
    with col2:
        utils.styled_metric("Avg K/D Ratio", f"{career_summary.get('K/D_Ratio', 0):.2f}")
    with col3:
        utils.styled_metric("Avg Hits / Game", f"{career_summary.get('Hits', 0):.2f}")
    with col4:
        utils.styled_metric("Avg Catches / Game", f"{career_summary.get('Catches', 0):.2f}")
    
    st.markdown("---")

    # --- Analysis Tabs ---
    tab1, tab2, tab3, tab4 = st.tabs(["Offense", "Defense", "Elimination Profile", "Performance Trends"])
    
    with tab1:
        st.subheader("Offensive Situational Analysis (Career Averages)")
        
        offensive_df = pd.DataFrame({
            'Situation': ['Singles', 'Multi-ball', 'Counters'],
            'Hits': [career_summary.get('Hits_Singles', 0), career_summary.get('Hits_Multi', 0), career_summary.get('Hits_Counters', 0)],
            'Throws': [career_summary.get('Throws_Singles', 0), career_summary.get('Throws_Multi', 0), career_summary.get('Throws_Counters', 0)]
        })
        offensive_df['Hit_Accuracy'] = (offensive_df['Hits'] / offensive_df['Throws'].replace(0, 1)) * 100

        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Bar(x=offensive_df['Situation'], y=offensive_df['Hits'], name='Avg Hits', marker_color='#1f77b4'), secondary_y=False)
        fig.add_trace(go.Bar(x=offensive_df['Situation'], y=offensive_df['Throws'], name='Avg Throws', marker_color='#aec7e8'), secondary_y=False)
        fig.add_trace(go.Scatter(x=offensive_df['Situation'], y=offensive_df['Hit_Accuracy'], name='Hit Accuracy (%)', mode='lines+markers', line=dict(color='red', width=3)), secondary_y=True)
        fig.update_layout(title_text='<b>Offensive Performance by Situation (Career Averages)</b>', barmode='group')
        fig.update_yaxes(title_text="Average Count", secondary_y=False)
        fig.update_yaxes(title_text="Hit Accuracy (%)", secondary_y=True, range=[0, 101])
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.subheader("Defensive Situational Analysis (Career Averages)")
        
        defensive_df = pd.DataFrame({
            'Situation': ['Singles', 'Multi-ball', 'Counters'],
            'Dodges': [career_summary.get('Dodges_Singles', 0), career_summary.get('Dodges_Multi', 0), career_summary.get('Dodges_Counters', 0)],
            'Blocks': [career_summary.get('Blocks_Singles', 0), career_summary.get('Blocks_Multi', 0), career_summary.get('Blocks_Counters', 0)],
            'Hit_Out': [career_summary.get('Out_Single_Hit', 0), career_summary.get('Out_Multi_Hit', 0), career_summary.get('Out_Counter_Hit', 0)]
        })

        fig2 = go.Figure()
        fig2.add_trace(go.Bar(x=defensive_df['Situation'], y=defensive_df['Dodges'], name='Avg Dodges', marker_color='#2ca02c'))
        fig2.add_trace(go.Bar(x=defensive_df['Situation'], y=defensive_df['Blocks'], name='Avg Blocks', marker_color='#98df8a'))
        fig2.add_trace(go.Bar(x=defensive_df['Situation'], y=defensive_df['Hit_Out'], name='Avg Times Hit Out', marker_color='#d62728'))
        fig2.update_layout(title_text='<b>Defensive Actions by Situation (Career Averages)</b>', barmode='group')
        st.plotly_chart(fig2, use_container_width=True)

    with tab3:
        st.subheader("Elimination Profile (Career Totals)")
        elimination_totals = player_all_games[['Out_Single_Hit', 'Out_Multi_Hit', 'Out_Counter_Hit', 'Caught_Out', 'Out_Other']].sum()
        elim_df = pd.DataFrame({
            'Reason': ['Hit (Single)', 'Hit (Multi)', 'Hit (Counter)', 'Caught Out', 'Other'],
            'Count': elimination_totals.values
        })
        elim_df = elim_df[elim_df['Count'] > 0]
        if not elim_df.empty:
            fig3 = px.pie(elim_df, values='Count', names='Reason', title='How Player is Eliminated (Career Total)')
            st.plotly_chart(fig3, use_container_width=True)
        else:
            st.info("This player has not been eliminated in the loaded games.")

    with tab4:
        st.subheader("Performance Trend Over Games")
        
        trend_metrics = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col]) and col not in ['Role_Cluster']]
        selected_trend_metric = st.selectbox("Select metric to track:", trend_metrics, index=trend_metrics.index('Overall_Performance'))
        
        if len(player_all_games) > 1:
            fig4 = px.line(
                player_all_games.sort_values('Game_ID'),
                x='Game_ID',
                y=selected_trend_metric,
                markers=True,
                title=f"{selected_trend_metric.replace('_', ' ')} Trend Across Games"
            )
            fig4.update_layout(xaxis_title="Game", yaxis_title=selected_trend_metric.replace('_', ' '))
            st.plotly_chart(fig4, use_container_width=True)
        else:
            st.info("Load more games to see performance trends over time.")

