import streamlit as st
import pandas as pd
import plotly.express as px
import utils

# --- Page Configuration and State Check ---
st.set_page_config(page_title="League Overview", page_icon="🏆", layout="wide")
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
st.header("🏆 Visual League Overview")
st.info(f"Analyzing detailed situational data from: **{st.session_state.source_name}**")
st.markdown("---")

col1, col2 = st.columns(2)

with col1:
    # --- NEW: Dynamic Leaderboard Chart ---
    st.subheader("Dynamic Leaderboard")
    metrics_list = [
        'Overall_Performance', 'K/D_Ratio', 'Hits', 'Catches', 'Dodges', 'Blocks',
        'Hits_Singles', 'Hits_Multi', 'Throws_Singles', 'Throws_Multi'
    ]
    selected_metric = st.selectbox("Select a metric to view leaders:", metrics_list)
    
    if selected_metric:
        leaderboard_df = df[['Player_ID', 'Team', selected_metric]].sort_values(by=selected_metric, ascending=False).head(10)
        
        fig = px.bar(
            leaderboard_df,
            x=selected_metric,
            y='Player_ID',
            orientation='h',
            color=selected_metric,
            color_continuous_scale='plasma',
            text_auto=True,
            title=f"Top 10 Players by {selected_metric.replace('_', ' ')}"
        )
        fig.update_layout(yaxis={'categoryorder':'total ascending'})
        st.plotly_chart(fig, use_container_width=True)

with col2:
    # --- NEW: Offensive vs. Defensive Rating Scatter Plot ---
    st.subheader("Offensive vs. Defensive Styles")
    
    fig2 = px.scatter(
        df,
        x="Offensive_Rating",
        y="Defensive_Rating",
        color="Team",
        hover_name="Player_ID",
        size='Overall_Performance',
        size_max=20,
        title="Player Styles: Offensive vs. Defensive Rating"
    )
    fig2.add_hline(y=df['Defensive_Rating'].mean(), line_dash="dot", annotation_text="League Avg. Defense")
    fig2.add_vline(x=df['Offensive_Rating'].mean(), line_dash="dot", annotation_text="League Avg. Offense")
    st.plotly_chart(fig2, use_container_width=True)

