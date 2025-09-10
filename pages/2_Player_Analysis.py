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
models = st.session_state.models

# --- Page Content ---
st.header("Individual Player Analysis")
st.info(f"Analyzing data from: **{st.session_state.source_name}**")
st.markdown("---")

player_list = sorted(df['Player_ID'].unique())
if not player_list:
    st.warning("No players found in the selected data source.")
    st.stop()
    
selected_player = st.selectbox("Select Player", player_list)

if selected_player:
    player_data = df[df['Player_ID'] == selected_player]
    
    # --- Player Summary Metrics ---
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        # --- CORRECTED: Calculate the average directly from the 'Overall_Performance' column ---
        player_career_avg = player_data['Overall_Performance'].mean()
        utils.styled_metric("Career Avg Performance", f"{player_career_avg:.2f}")
    with col2:
        player_kd_ratio = player_data['K/D_Ratio'].mean()
        utils.styled_metric("Avg K/D Ratio", f"{player_kd_ratio:.2f}")
    with col3:
        player_hit_accuracy = player_data['Hit_Accuracy'].mean()
        utils.styled_metric("Career Hit Accuracy", f"{player_hit_accuracy:.1%}")
    with col4:
        player_role = player_data['Player_Role'].iloc[0] if 'Player_Role' in player_data.columns and not player_data['Player_Role'].empty else "N/A"
        utils.styled_metric("Assigned Player Role", player_role)
        
    st.markdown("---")
    
    # --- Visualizations ---
    st.subheader("Player Dashboard")
    fig = utils.create_player_dashboard(df, selected_player)
    if fig:
        st.plotly_chart(fig, use_container_width=True)

    # --- Performance and Stamina Trends ---
    st.subheader("Performance Trends")
    col1, col2 = st.columns(2)
    with col1:
        st.write("**Performance Across Matches**")
        improvement_fig = utils.create_improvement_chart(df, selected_player)
        if improvement_fig:
            st.plotly_chart(improvement_fig, use_container_width=True)
        else:
            st.info("Not enough matches played to analyze improvement trends.")
    with col2:
        st.write("**Performance Within Matches (Stamina)**")
        stamina_fig = utils.create_stamina_chart(df, selected_player)
        if stamina_fig:
            st.plotly_chart(stamina_fig, use_container_width=True)
        else:
            st.info("Not enough games played to analyze stamina trends.")
            
    # --- Elimination Profile ---
    st.subheader("Elimination Profile")
    elim_col1, elim_col2 = st.columns(2)
    with elim_col1:
        hit_out_total = player_data['Hit_Out'].sum()
        utils.styled_metric("Eliminated by Hit", f"{hit_out_total:.0f}")
    with elim_col2:
        caught_out_total = player_data['Caught_Out'].sum()
        utils.styled_metric("Eliminated by Catch", f"{caught_out_total:.0f}")

    if caught_out_total > (hit_out_total / 3) and caught_out_total > 2:
        st.warning("🎯 **Coaching Insight:** This player's throws get caught relatively often. Focus on shot selection and avoiding risky throws to strong catchers.")
    elif hit_out_total > (caught_out_total * 2) and hit_out_total > 2:
        st.warning("🏃 **Coaching Insight:** This player is eliminated by being hit directly much more often than being caught. Focus on dodging drills and improving spatial awareness.")
    else:
        st.info("✅ **Insight:** This player has a balanced elimination profile.")

    st.markdown("---")
    
    # --- Player Comparison ---
    st.subheader("Player Comparison")
    if st.checkbox("Enable Player Comparison"):
        comparison_player = st.selectbox(
            "Compare with:",
            [p for p in player_list if p != selected_player],
            key="comparison_player_select"
        )
        if comparison_player:
            col1, col2 = st.columns(2)
            with col1:
                fig1 = utils.create_player_dashboard(df, selected_player)
                if fig1:
                    st.plotly_chart(fig1, use_container_width=True, key="compare_chart_1")
            with col2:
                fig2 = utils.create_player_dashboard(df, comparison_player)
                if fig2:
                    st.plotly_chart(fig2, use_container_width=True, key="compare_chart_2")
