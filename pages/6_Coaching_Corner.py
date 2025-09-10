import streamlit as st
import utils

# --- Page Configuration and State Check ---
st.set_page_config(page_title="Coaching Corner", page_icon="🧑‍🏫", layout="wide")
st.markdown(utils.load_css(), unsafe_allow_html=True)

if 'data_loaded' not in st.session_state or not st.session_state.data_loaded:
    st.warning("Please select and load a data source from the 🏠 Home page first.")
    st.stop()

df = st.session_state.df_enhanced
models = st.session_state.models

# --- Page Content ---
st.header("🧑‍🏫 Coaching Corner")
st.info(f"Generating specific, data-driven advice from: **{st.session_state.source_name}**")
st.markdown("---")

coach_mode = st.radio("Select Coaching Mode", ["Player Coaching", "Team Coaching"], horizontal=True)

if coach_mode == "Player Coaching":
    player_list = sorted(df['Player_ID'].unique())
    if not player_list:
        st.warning("No players found to coach.")
        st.stop()
        
    selected_player = st.selectbox("Select a Player to Coach", player_list)
    if selected_player:
        # This function now generates a more detailed report in utils.py
        report, fig = utils.generate_player_coaching_report(df, selected_player)
        
        st.markdown('<div class="insight-box">', unsafe_allow_html=True)
        for line in report:
            st.markdown(line)
        st.markdown('</div>', unsafe_allow_html=True)
        
        if fig:
            st.plotly_chart(fig, use_container_width=True)

elif coach_mode == "Team Coaching":
    team_list = sorted(df['Team'].unique())
    if not team_list:
        st.warning("No teams found to coach.")
        st.stop()
        
    selected_team = st.selectbox("Select a Team to Coach", team_list)
    if selected_team:
        report = utils.generate_team_coaching_report(df, selected_team)
        
        st.markdown('<div class="insight-box">', unsafe_allow_html=True)
        for line in report:
            st.markdown(line)
        st.markdown('</div>', unsafe_allow_html=True)
