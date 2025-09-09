import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import utils

# --- Page Configuration and State Check ---
st.set_page_config(page_title="Player Analysis", page_icon="👤", layout="wide")
st.markdown(utils.load_css(), unsafe_allow_html=True)

# Check if data has been loaded from the Home page
if 'data_loaded' not in st.session_state or not st.session_state.data_loaded:
    st.warning("Please select and load a data source from the 🏠 Home page first.")
    st.stop()

# If data is loaded, get it from session state
df = st.session_state.df_enhanced
models = st.session_state.models

# --- Page Content ---
st.header("Individual Player Analysis")
st.info(f"Analyzing data from: **{st.session_state.source_name}**")
st.markdown("---")

# --- Player Selection ---
# Use a placeholder if the player list is empty to prevent errors
player_list = sorted(df['Player_ID'].unique())
if not player_list:
    st.warning("No players found in the selected data source.")
    st.stop()
    
selected_player = st.selectbox("Select Player", player_list)

if selected_player:
    # --- Defines the player's data once to be used by all sections below ---
    player_data = df[df['Player_ID'] == selected_player]
    player_summary_data = player_data.iloc[0]
    player_career_avg = player_summary_data['Avg_Performance']

    # --- Key Metrics ---
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        utils.styled_metric("Player Role", player_summary_data.get('Player_Role', 'N/A'))
    with col2:
        utils.styled_metric("Win Rate", f"{player_summary_data['Win_Rate']:.1%}")
    with col3:
        utils.styled_metric("Avg K/D Ratio", f"{player_summary_data['Avg_KD_Ratio']:.2f}",
                            help_text="Ratio of opponents eliminated to times this player was eliminated.")
    with col4:
        utils.styled_metric("Avg Net Impact", f"{player_summary_data['Avg_Net_Impact']:.2f}",
                            help_text="(Hits + Catches) - Times Eliminated.")

    # --- Main Visual ---
    fig = utils.create_player_dashboard(df, selected_player)
    if fig:
        st.plotly_chart(fig, width='stretch')
    
    st.markdown("---")

    # --- Metric Trend Over Time ---
    st.subheader("Metric Trend Over Time")

    # Define the list of stats the user can choose from
    stats_options = [
        'Overall_Performance',
        'Hits',
        'Catches',
        'Dodges',
        'K/D_Ratio',
        'Net_Impact',
        'Hit_Accuracy',
        'Defensive_Efficiency',
        'Offensive_Rating',
        'Defensive_Rating'
    ]

    # Create the dropdown menu
    selected_stat = st.selectbox(
        "Select a metric to track over time:",
        stats_options,
        help="Choose any statistic to see how it has trended across all matches played."
    )

    # Pass the user's selection to the chart function
    if selected_stat:
        improvement_fig = utils.create_improvement_chart(df, selected_player, metric_to_plot=selected_stat)
        
        if improvement_fig:
            st.plotly_chart(improvement_fig, width='stretch')
        else:
            st.info("Not enough data (requires participation in at least 2 matches) to show a trend for this player.")

    st.markdown("---")
    
    # --- Performance in Specific Match ---
    st.subheader("Performance in a Specific Match")
    player_matches = sorted(player_data['Match_ID'].dropna().unique())
    if player_matches:
        selected_match = st.selectbox("Select a match to analyze performance:", player_matches)
        
        if selected_match:
            match_perf_df = player_data[player_data['Match_ID'] == selected_match]
            avg_perf_in_match = match_perf_df['Overall_Performance'].mean()
            
            st.metric(
                label=f"Avg Performance in {selected_match}",
                value=f"{avg_perf_in_match:.2f}",
                delta=f"{avg_perf_in_match - player_career_avg:.2f} vs. Career Average"
            )
    else:
        st.info("This player has not played in any matches.")

    st.markdown("---")

    # --- Stamina Analysis Section ---
    st.subheader("Stamina Analysis")
    stamina_fig = utils.create_stamina_chart(df, selected_player)
    if stamina_fig:
        stamina_trend = player_data['Stamina_Trend'].iloc[0]
        st.metric("Stamina Trend Score", f"{stamina_trend:.2f}", help="A positive score means the player's performance tends to improve in later games of a match. A negative score means it tends to drop.")
        st.plotly_chart(stamina_fig, width='stretch')
    else:
        st.info("Not enough data across multiple matches to analyze this player's stamina.")
    
    st.markdown("---")

    # --- Elimination Profile Section ---
    st.subheader("Elimination Profile")
    col1, col2 = st.columns([1, 2])

    with col1:
        hit_out_total = player_summary_data['Total_Hit_Out']
        caught_out_total = player_summary_data['Total_Caught_Out']
        
        if hit_out_total + caught_out_total == 0:
            st.info("This player has not been eliminated yet.")
        else:
            elimination_labels = ['Hit Out', 'Caught Out']
            elimination_values = [hit_out_total, caught_out_total]
            
            fig_pie = go.Figure(data=[go.Pie(
                labels=elimination_labels, values=elimination_values,
                hole=.3, marker_colors=['#FF6B6B', '#FECA57']
            )])
            fig_pie.update_layout(title_text="Breakdown of Eliminations", showlegend=False)
            st.plotly_chart(fig_pie, use_container_width=True)
    
    with col2:
        st.write(f"Analyzing how **{selected_player}** gets eliminated provides insight into their defensive vulnerabilities.")
        st.metric("Total Times Eliminated by Being Hit", f"{hit_out_total:.0f}")
        st.metric("Total Times Eliminated by a Catch", f"{caught_out_total:.0f}")

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