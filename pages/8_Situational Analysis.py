import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import utils

# --- Page Configuration and State Check ---
st.set_page_config(page_title="Situational Analysis", page_icon="??", layout="wide")
st.markdown(utils.load_css(), unsafe_allow_html=True)

if 'data_loaded' not in st.session_state or not st.session_state.data_loaded:
    st.warning("Please select and load a data source from the ?? Home page first.")
    st.stop()

df = st.session_state.df_enhanced

# --- Page Content ---
st.header("?? Situational Performance Analysis")
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

    # --- Data Cleaning and Conversion ---
    # Convert percentage strings (e.g., "50.00%") to floats for calculations
    for col in df.columns:
        if 'Survivability' in col and df[col].dtype == 'object':
            df[col] = df[col].str.replace('%', '').astype(float) / 100.0

    # --- Analysis Section ---
    tab1, tab2, tab3 = st.tabs(["Offensive Breakdown", "Defensive Breakdown", "Elimination Profile"])

    with tab1:
        st.subheader("Offensive Breakdown by Situation")
        
        # Data for Offensive Chart
        offensive_data = {
            'Situation': ['Singles', 'Multi-ball', 'Counters', 'Pres'],
            'Hits': [player_data.get('Hits_Singles', 0), player_data.get('Hits_Multi', 0), player_data.get('Hits_Counters', 0), player_data.get('Hits_Pres', 0)],
            'Throws': [player_data.get('Throws_Singles', 0), player_data.get('Throws_Multi', 0), player_data.get('Throws_Counters', 0), player_data.get('Throws_Pres', 0)]
        }
        offensive_df = pd.DataFrame(offensive_data)
        
        # Calculate Hit Accuracy for each situation
        offensive_df['Hit_Accuracy'] = (offensive_df['Hits'] / offensive_df['Throws'].replace(0, 1)) * 100

        fig_offense = make_subplots(specs=[[{"secondary_y": True}]])

        # Add bars for Hits and Throws
        fig_offense.add_trace(go.Bar(x=offensive_df['Situation'], y=offensive_df['Hits'], name='Hits', marker_color='#1f77b4'), secondary_y=False)
        fig_offense.add_trace(go.Bar(x=offensive_df['Situation'], y=offensive_df['Throws'], name='Throws', marker_color='#aec7e8'), secondary_y=False)

        # Add line for Hit Accuracy
        fig_offense.add_trace(go.Scatter(x=offensive_df['Situation'], y=offensive_df['Hit_Accuracy'], name='Hit Accuracy (%)', mode='lines+markers', marker_color='#ff7f0e'), secondary_y=True)

        fig_offense.update_layout(title_text='Hits, Throws, and Accuracy by Situation', barmode='group')
        fig_offense.update_yaxes(title_text="Count (Hits & Throws)", secondary_y=False)
        fig_offense.update_yaxes(title_text="Hit Accuracy (%)", secondary_y=True, range=[0, 100])
        
        st.plotly_chart(fig_offense, use_container_width=True)

    with tab2:
        st.subheader("Defensive Breakdown and Survivability")

        # Data for Defensive Chart
        defensive_data = {
            'Situation': ['Singles', 'Multi-ball', 'Counters', 'Pres'],
            'Dodges': [player_data.get('Dodges_Singles', 0), player_data.get('Dodges_Multi', 0), player_data.get('Dodges_Counters', 0), player_data.get('Dodges_Pres', 0)],
            'Blocks': [player_data.get('Blocks_Singles', 0), player_data.get('Blocks_Multi', 0), player_data.get('Blocks_Counters', 0), player_data.get('Blocks_Pres', 0)],
            'Thrown_At': [player_data.get('Thrown_At_Singles', 0), player_data.get('Thrown_At_Multi', 0), player_data.get('Thrown_At_Counters', 0), player_data.get('Thrown_At_Pres', 0)],
            'Survivability': [player_data.get('Survivability_Singles', 0), player_data.get('Survivability_Multi', 0), player_data.get('Survivability_Counters', 0), player_data.get('Survivability_Pres', 0)]
        }
        defensive_df = pd.DataFrame(defensive_data)

        fig_defense = make_subplots(specs=[[{"secondary_y": True}]])

        # Add bars for Dodges, Blocks, and Times Thrown At
        fig_defense.add_trace(go.Bar(x=defensive_df['Situation'], y=defensive_df['Dodges'], name='Dodges', marker_color='#2ca02c'), secondary_y=False)
        fig_defense.add_trace(go.Bar(x=defensive_df['Situation'], y=defensive_df['Blocks'], name='Blocks', marker_color='#98df8a'), secondary_y=False)
        fig_defense.add_trace(go.Bar(x=defensive_df['Situation'], y=defensive_df['Thrown_At'], name='Times Thrown At', marker_color='#d62728'), secondary_y=False)

        # Add line for Survivability
        fig_defense.add_trace(go.Scatter(x=defensive_df['Situation'], y=defensive_df['Survivability'], name='Survivability %', mode='lines+markers', marker_color='#ff9896'), secondary_y=True)
        
        fig_defense.update_layout(title_text='Defensive Actions and Survivability by Situation', barmode='group')
        fig_defense.update_yaxes(title_text="Count (Actions)", secondary_y=False)
        fig_defense.update_yaxes(title_text="Survivability (%)", secondary_y=True, range=[0, 1])

        st.plotly_chart(fig_defense, use_container_width=True)

    with tab3:
        st.subheader("Elimination Profile")
        
        elimination_data = {
            'Reason': ['Hit (Single)', 'Hit (Multi)', 'Hit (Counter)', 'Hit (Pre)', 'Caught', 'Other'],
            'Count': [
                player_data.get('Out_Single_Hit', 0),
                player_data.get('Out_Multi_Hit', 0),
                player_data.get('Out_Counter_Hit', 0),
                player_data.get('Out_Pre_Hit', 0),
                player_data.get('Caught_Out', 0),
                player_data.get('Out_Other', 0)
            ]
        }
        elimination_df = pd.DataFrame(elimination_data)
        elimination_df = elimination_df[elimination_df['Count'] > 0] # Filter out zero-count reasons

        if not elimination_df.empty:
            fig_elim = px.pie(elimination_df, values='Count', names='Reason', title='How Player is Eliminated')
            st.plotly_chart(fig_elim, use_container_width=True)
        else:
            st.info("This player was not eliminated in the selected dataset.")
