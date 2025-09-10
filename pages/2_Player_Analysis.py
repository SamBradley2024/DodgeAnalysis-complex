import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import utils
import plotly.express as px

# --- Page Configuration and State Check ---
st.set_page_config(page_title="Player Analysis", page_icon="👤", layout="wide")
st.markdown(utils.load_css(), unsafe_allow_html=True)

if 'data_loaded' not in st.session_state or not st.session_state.data_loaded:
    st.warning("Please select and load a data source from the 🏠 Home page first.")
    st.stop()

df = st.session_state.df_enhanced

# --- Data Cleaning ---
for col in df.columns:
    if df[col].dtype == 'object' and col not in ['Player_ID', 'Team', 'Match_ID', 'Game_ID', 'Game_Outcome', 'Player_Role']:
        df[col] = pd.to_numeric(df[col].str.replace('%', '', regex=False).str.replace('#DIV/0!', '0', regex=False), errors='coerce').fillna(0)
        if '%' in str(df[col].iloc[0]):
             df[col] = df[col] / 100.0

# --- Page Content ---
st.header("👤 Individual Player Analysis")
st.info(f"Analyzing detailed situational data from: **{st.session_state.source_name}**")

player_list = sorted(df['Player_ID'].unique())
selected_player = st.selectbox("Select Player", player_list)

if selected_player:
    player_data = df[df['Player_ID'] == selected_player]
    player_stats = player_data.iloc[0]

    # --- NEW: Offensive Analysis with Stacked Chart ---
    st.subheader("Offensive Breakdown")
    
    offensive_df = pd.DataFrame({
        'Situation': ['Singles', 'Multi-ball', 'Counters', 'Pres'],
        'Hits': [player_stats.get(c, 0) for c in ['Hits_Singles', 'Hits_Multi', 'Hits_Counters', 'Hits_Pres']],
        'Throws': [player_stats.get(c, 0) for c in ['Throws_Singles', 'Throws_Multi', 'Throws_Counters', 'Throws_Pres']]
    })
    offensive_df['Misses'] = offensive_df['Throws'] - offensive_df['Hits']
    offensive_df['Hit_Accuracy'] = (offensive_df['Hits'] / offensive_df['Throws'].replace(0, 1))

    fig_offense = go.Figure()
    fig_offense.add_trace(go.Bar(x=offensive_df['Situation'], y=offensive_df['Hits'], name='Hits', marker_color='green'))
    fig_offense.add_trace(go.Bar(x=offensive_df['Situation'], y=offensive_df['Misses'], name='Misses', marker_color='red'))
    fig_offense.update_layout(barmode='stack', title_text='<b>Offensive Performance: Hits vs. Misses</b>', yaxis_title='Total Throws')
    
    # Add accuracy annotations
    for i, row in offensive_df.iterrows():
        fig_offense.add_annotation(x=row['Situation'], y=row['Throws'], text=f"<b>{row['Hit_Accuracy']:.1%} Acc.</b>", showarrow=False, yshift=10)

    st.plotly_chart(fig_offense, use_container_width=True)

    # --- NEW: Defensive Analysis with Stacked Chart ---
    st.subheader("Defensive Breakdown")
    
    defensive_df = pd.DataFrame({
        'Situation': ['Singles', 'Multi-ball', 'Counters', 'Pres'],
        'Dodges': [player_stats.get(c, 0) for c in ['Dodges_Singles', 'Dodges_Multi', 'Dodges_Counters', 'Dodges_Pres']],
        'Blocks': [player_stats.get(c, 0) for c in ['Blocks_Singles', 'Blocks_Multi', 'Blocks_Counters', 'Blocks_Pres']],
        'Times_Thrown_At': [player_stats.get(c, 0) for c in ['Thrown_At_Singles', 'Thrown_At_Multi', 'Thrown_At_Counters', 'Thrown_At_Pres']],
    })
    defensive_df['Hit_Out'] = defensive_df['Times_Thrown_At'] - defensive_df['Dodges'] - defensive_df['Blocks']
    defensive_df['Survivability'] = 1 - (defensive_df['Hit_Out'] / defensive_df['Times_Thrown_At'].replace(0, 1))

    fig_defense = go.Figure()
    fig_defense.add_trace(go.Bar(x=defensive_df['Situation'], y=defensive_df['Dodges'], name='Dodges', marker_color='blue'))
    fig_defense.add_trace(go.Bar(x=defensive_df['Situation'], y=defensive_df['Blocks'], name='Blocks', marker_color='lightblue'))
    fig_defense.add_trace(go.Bar(x=defensive_df['Situation'], y=defensive_df['Hit_Out'], name='Hit Out', marker_color='orange'))
    fig_defense.update_layout(barmode='stack', title_text='<b>Defensive Performance: Actions vs. Hit Out</b>', yaxis_title='Times Thrown At')
    
    # Add survivability annotations
    for i, row in defensive_df.iterrows():
        fig_defense.add_annotation(x=row['Situation'], y=row['Times_Thrown_At'], text=f"<b>{row['Survivability']:.1%} Surv.</b>", showarrow=False, yshift=10)

    st.plotly_chart(fig_defense, use_container_width=True)

    # --- NEW: Elimination Profile (Moved from Situational) ---
    st.subheader("Elimination Profile")
    col1, col2 = st.columns(2)
    with col1:
        elim_data = {
            'Reason': ['Hit by Player', 'Caught by Opponent'],
            'Count': [player_stats.get('Hit_Out', 0), player_stats.get('Caught_Out', 0)]
        }
        elim_df = pd.DataFrame(elim_data)
        fig_elim = px.pie(elim_df, values='Count', names='Reason', title='How Player is Eliminated')
        st.plotly_chart(fig_elim, use_container_width=True)
    with col2:
        catch_data = {
            'Result': ['Successful Catches', 'Failed Attempts'],
            'Count': [player_stats.get('Catches', 0), player_stats.get('Catches_Attempted', 0) - player_stats.get('Catches', 0)]
        }
        catch_df = pd.DataFrame(catch_data)
        fig_catch = px.pie(catch_df, values='Count', names='Result', title='Player Catching Performance')
        st.plotly_chart(fig_catch, use_container_width=True)

    # --- RESTORED: Performance Over Time ---
    st.subheader("Performance Trend")
    st.markdown("This chart shows the player's overall performance score across all recorded games.")
    fig_trend = px.line(player_data, x='Game_ID', y='Overall_Performance', markers=True, title=f'{selected_player} - Performance Per Game')
    # If there's only one data point, show it as a dot
    if len(player_data) == 1:
        fig_trend.update_traces(mode='markers', marker=dict(size=12))
    st.plotly_chart(fig_trend, use_container_width=True)

