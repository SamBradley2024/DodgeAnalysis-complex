import streamlit as st
import utils

# --- Page Configuration and State Check ---
st.set_page_config(page_title="AI Insights", page_icon="🤖", layout="wide")
st.markdown(utils.load_css(), unsafe_allow_html=True)

if 'data_loaded' not in st.session_state or not st.session_state.data_loaded:
    st.warning("Please select and load a data source from the 🏠 Home page first.")
    st.stop()

df = st.session_state.df_enhanced

# --- Page Content ---
st.header("🤖 AI-Generated Insights")
st.info(f"This section provides automated insights discovered from the detailed situational data in **{st.session_state.source_name}**.")
st.markdown("---")

# --- Generated Insights Section ---
st.subheader(
    "Generated Strategic & Situational Insights",
    help="This section displays automated insights discovered by analyzing the entire dataset. It identifies top situational performers and potential strategic tendencies."
)

# This function is now updated in utils.py to find situational patterns
insights = utils.generate_situational_insights(df)

if not insights:
    st.warning("Could not generate any advanced insights. The dataset may be too small or lack sufficient variation for the AI models to find patterns.")
else:
    for insight in insights:
        st.markdown(f'<div class="insight-box">{insight}</div>', unsafe_allow_html=True)
