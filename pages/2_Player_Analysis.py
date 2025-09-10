import streamlit as st
import pandas as pd
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

# --- Data Cleaning ---
# This ensures all numeric columns are correctly typed for calculations.
for col in df.columns:
    if df[col].dtype == 'object' and col not in ['Player_ID', 'Team', 'Match_ID', 'Game_ID', 'Game_Outcome', 'Player_Role']:
        if '%' in str(df[col].iloc[0]):
            df[col] = df[col].str.replace('#DIV/0!', '0', regex=False)
            df[col] = pd.to_numeric(df[col].str.replace('%', '', regex=False), errors='coerce').fillna(0) / 100.0
        else:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

# --- Page Content ---
st.header("👤 Individual Player Analysis")
st.info(f"Analyzing detailed situational data from: **{st.session_state.source_name}**")
st.markdown("---")

# --- Player Selection ---
player_list = sorted(df['Player_ID'].unique())
if not player_list:
    st.warning("No players found in the selected data source.")
    st.stop()
    
selected_player = st.selectbox("Select Player", player_list)

if selected_player:
    player_data = df[df['Player_ID'] == selected_player].iloc[0]
    
    # --- Key Performance Indicators ---
    st.subheader("Key Performance Indicators")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        utils.styled_metric("Overall Performance", f"{player_data.get('Overall_Performance', 0):.2f}")
    with col2:
        utils.styled_metric("K/D Ratio", f"{player_data.get('K/D_Ratio', 0):.2f}")
    with col3:
        utils.styled_metric("Total Hits", f"{player_data.get('Hits', 0):.0f}")
    with col4:
        utils.styled_metric("Total Catches", f"{player_data.get('Catches', 0):.0f}")

    st.markdown("---")
    
    # --- Situational Dashboards ---
    st.subheader("Situational Performance Dashboard")
    
    # --- Offensive Chart ---
    offensive_data = {
        'Situation': ['Singles', 'Multi-ball', 'Counters', 'Pres'],
        'Hits': [player_data.get('Hits_Singles', 0), player_data.get('Hits_Multi', 0), player_data.get('Hits_Counters', 0), player_data.get('Hits_Pres', 0)],
        'Throws': [player_data.get('Throws_Singles', 0), player_data.get('Throws_Multi', 0), player_data.get('Throws_Counters', 0), player_data.get('Throws_Pres', 0)]
    }
    offensive_df = pd.DataFrame(offensive_data)
    offensive_df['Hit_Accuracy'] = (offensive_df['Hits'] / offensive_df['Throws'].replace(0, 1)) * 100
    
    fig_offense = make_subplots(specs=[[{"secondary_y": True}]])
    fig_offense.add_trace(go.Bar(x=offensive_df['Situation'], y=offensive_df['Hits'], name='Hits'), secondary_y=False)
    fig_offense.add_trace(go.Bar(x=offensive_df['Situation'], y=offensive_df['Throws'], name='Throws'), secondary_y=False)
    fig_offense.add_trace(go.Scatter(x=offensive_df['Situation'], y=offensive_df['Hit_Accuracy'], name='Hit Accuracy (%)', mode='lines+markers'), secondary_y=True)
    fig_offense.update_layout(title_text='<b>Offensive Actions by Situation</b>', barmode='group')
    fig_offense.update_yaxes(title_text="Count", secondary_y=False)
    fig_offense.update_yaxes(title_text="Accuracy (%)", secondary_y=True, range=[0, 100])
    
    # --- Defensive Chart ---
    defensive_data = {
        'Situation': ['Singles', 'Multi-ball', 'Counters', 'Pres'],
        'Dodges': [player_data.get('Dodges_Singles', 0), player_data.get('Dodges_Multi', 0), player_data.get('Dodges_Counters', 0), player_data.get('Dodges_Pres', 0)],
        'Blocks': [player_data.get('Blocks_Singles', 0), player_data.get('Blocks_Multi', 0), player_data.get('Blocks_Counters', 0), player_data.get('Blocks_Pres', 0)],
        'Survivability': [player_data.get('Survivability_Singles', 0), player_data.get('Survivability_Multi', 0), player_data.get('Survivability_Counters', 0), player_data.get('Survivability_Pres', 0)]
    }
    defensive_df = pd.DataFrame(defensive_data)

    fig_defense = make_subplots(specs=[[{"secondary_y": True}]])
    fig_defense.add_trace(go.Bar(x=defensive_df['Situation'], y=defensive_df['Dodges'], name='Dodges'), secondary_y=False)
    fig_defense.add_trace(go.Bar(x=defensive_df['Situation'], y=defensive_df['Blocks'], name='Blocks'), secondary_y=False)
    fig_defense.add_trace(go.Scatter(x=defensive_df['Situation'], y=defensive_df['Survivability'], name='Survivability %', mode='lines+markers'), secondary_y=True)
    fig_defense.update_layout(title_text='<b>Defensive Actions by Situation</b>', barmode='group')
    fig_defense.update_yaxes(title_text="Count", secondary_y=False)
    fig_defense.update_yaxes(title_text="Survivability (%)", secondary_y=True, range=[0, 1])

    # Display charts side-by-side
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(fig_offense, use_container_width=True)
    with col2:
        st.plotly_chart(fig_defense, use_container_width=True)
