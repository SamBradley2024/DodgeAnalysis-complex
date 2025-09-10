import streamlit as st
import pandas as pd
import plotly.express as px
import utils

# --- Page Configuration and State Check ---
st.set_page_config(page_title="Match Analysis", page_icon="⚔️", layout="wide")
st.markdown(utils.load_css(), unsafe_allow_html=True)

if 'data_loaded' not in st.session_state or not st.session_state.data_loaded:
    st.warning("Please select and load a data source from the 🏠 Home page first.")
    st.stop()

df = st.session_state.df_enhanced

# --- Data Cleaning ---
for col in df.columns:
    if df[col].dtype == 'object' and col not in ['Player_ID', 'Team', 'Match_ID', 'Game_ID', 'Game_Outcome', 'Player_Role']:
        if '%' in str(df[col].iloc[0]):
            df[col] = df[col].str.replace('#DIV/0!', '0', regex=False)
            df[col] = pd.to_numeric(df[col].str.replace('%', '', regex=False), errors='coerce').fillna(0) / 100.0
        else:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

# --- Page Content ---
st.header("⚔️ Match Analysis")
st.info(f"Analyzing detailed match data from: **{st.session_state.source_name}**")

match_list = sorted(df['Match_ID'].unique())
if not match_list:
    st.warning("No matches found in the selected data source.")
    st.stop()

selected_match = st.selectbox("Select a Match to Analyze", match_list)

if selected_match:
    match_df = df[df['Match_ID'] == selected_match].copy()
    
    st.markdown("---")
    st.subheader(f"Situational Breakdown for {selected_match}")

    # --- Situational Heatmap ---
    # Define the situational columns we want to show
    situational_cols = [
        'Hits_Singles', 'Hits_Multi', 'Hits_Counters',
        'Throws_Singles', 'Throws_Multi', 'Throws_Counters',
        'Dodges_Singles', 'Dodges_Multi', 'Dodges_Counters',
        'Blocks_Singles', 'Blocks_Multi', 'Blocks_Counters'
    ]
    
    # Make sure the columns exist in the dataframe
    existing_cols = ['Player_ID', 'Team'] + [col for col in situational_cols if col in match_df.columns]
    
    heatmap_data = match_df[existing_cols].set_index('Player_ID')
    
    # We only want to display the numeric situational columns in the heatmap
    heatmap_numeric_cols = [col for col in existing_cols if col not in ['Player_ID', 'Team']]

    if not heatmap_numeric_cols:
        st.warning("No detailed situational data available for this match.")
    else:
        fig = px.imshow(
            heatmap_data[heatmap_numeric_cols],
            text_auto=True,
            aspect="auto",
            color_continuous_scale='Viridis',
            labels=dict(x="Situation", y="Player", color="Total Actions"),
            title=f"<b>Player Actions Heatmap for {selected_match}</b>"
        )
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    
    # --- Overall Match Performance Table ---
    st.subheader(f"Overall Player Performance in {selected_match}")
    
    # Select and rename columns for a clean display
    summary_df = match_df[[
        'Player_ID', 'Team', 'Overall_Performance', 'K/D_Ratio', 
        'Hits', 'Throws', 'Catches', 'Dodges', 'Blocks', 'Times_Eliminated'
    ]].sort_values('Overall_Performance', ascending=False)
    
    st.dataframe(summary_df, use_container_width=True)
