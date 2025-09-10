import streamlit as st
import utils

# --- Page Configuration and State Check ---
st.set_page_config(page_title="Coaching Corner", page_icon="🧑‍🏫", layout="wide")
st.markdown(utils.load_css(), unsafe_allow_html=True)

if 'data_loaded' not in st.session_state or not st.session_state.data_loaded:
    st.warning("Please select and load a data source from the 🏠 Home page first.")
    st.stop()

df = st.session_state.df_enhanced

# --- Page Content ---
st.header("🧑‍🏫 Advanced Coaching Corner")
st.info(f"Generating advanced, data-driven advice from: **{st.session_state.source_name}**")
st.markdown("---")

player_list = sorted(df['Player_ID'].unique())
selected_player = st.selectbox("Select a Player to Coach", player_list)

if selected_player:
    # This function now returns data for two separate charts
    strengths, weaknesses, fig_league, fig_role = utils.generate_advanced_coaching_report(df, selected_player)
    
    # --- Strengths and Weaknesses Display ---
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("✅ Top Strengths")
        st.markdown('<div class="insight-box" style="border-left-color: #2ca02c;">', unsafe_allow_html=True)
        for line in strengths:
            st.markdown(line)
        st.markdown('</div>', unsafe_allow_html=True)
    with col2:
        st.subheader("⚠️ Areas for Improvement")
        st.markdown('<div class="warning-box" style="border-left-color: #d62728;">', unsafe_allow_html=True)
        for line in weaknesses:
            st.markdown(line)
        st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    # --- NEW: Larger, Vertical Chart Layout ---
    st.subheader("Player Percentile Rankings vs. League Average")
    st.plotly_chart(fig_league, use_container_width=True)

    if fig_role:
        st.subheader("Player Stats vs. Role Average")
        st.plotly_chart(fig_role, use_container_width=True)

