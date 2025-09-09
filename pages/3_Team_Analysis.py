import streamlit as st
import utils 

# --- State Management and Sidebar ---
st.set_page_config(page_title="Team Analysis", page_icon="🏆", layout="wide")
st.markdown(utils.load_css(), unsafe_allow_html=True)

# Check if data has been loaded from the Home page
if 'data_loaded' not in st.session_state or not st.session_state.data_loaded:
    st.warning("Please select and load a data source from the 🏠 Home page first.")
    st.stop()

# If data is loaded, get it from session state
df = st.session_state.df_enhanced
models = st.session_state.models

# --- Page Content ---
st.header("Team Analysis")
st.info(f"Analyzing data from: **{st.session_state.source_name}**")
st.markdown("---")



team_list = sorted(df['Team'].unique())
if not team_list:
    st.warning("No teams found in the selected worksheet.")
    st.stop()

selected_team = st.selectbox("Select Team", team_list)

if selected_team:
    team_data = df[df['Team'] == selected_team]
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        utils.styled_metric("Team Size", team_data['Player_ID'].nunique())
    with col2:
        win_rate = (team_data['Game_Outcome'] == 'Win').mean()
        utils.styled_metric("Team Win Rate", f"{win_rate:.1%}")
    with col3:
        avg_perf = team_data['Overall_Performance'].mean()
        utils.styled_metric("Avg Performance", f"{avg_perf:.2f}")
    with col4:
        if 'Player_Role' in team_data.columns and not team_data['Player_Role'].dropna().empty:
            dominant_role = team_data['Player_Role'].mode()[0]
            utils.styled_metric("Dominant Role", dominant_role)
        else:
            utils.styled_metric("Dominant Role", "N/A")

    fig = utils.create_team_analytics(df, selected_team)
    st.plotly_chart(fig, use_container_width=True)