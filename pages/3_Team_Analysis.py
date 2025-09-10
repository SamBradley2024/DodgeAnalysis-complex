import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import utils

# --- Page Configuration and State Check ---
st.set_page_config(page_title="Team Analysis", page_icon="🏆", layout="wide")
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
st.header("🏆 Team Analysis")
st.info(f"Analyzing detailed situational data from: **{st.session_state.source_name}**")
st.markdown("---")

team_list = sorted(df['Team'].unique())
if not team_list:
    st.warning("No teams found in the selected data source.")
    st.stop()

selected_team = st.selectbox("Select Team", team_list)

if selected_team:
    team_data = df[df['Team'] == selected_team]
    
    # --- Key Team Metrics ---
    st.subheader("Team Performance Summary")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        utils.styled_metric("Team Size", team_data['Player_ID'].nunique())
    with col2:
        utils.styled_metric("Total Hits", f"{team_data['Hits'].sum():.0f}")
    with col3:
        utils.styled_metric("Total Catches", f"{team_data['Catches'].sum():.0f}")
    with col4:
        utils.styled_metric("Avg. Team Performance", f"{team_data['Overall_Performance'].mean():.2f}")

    st.markdown("---")

    # --- New Situational Performance Chart ---
    st.subheader("Team Situational Performance")
    
    # Aggregate situational stats for the whole team
    situational_totals = {
        'Situation': ['Singles', 'Multi-ball', 'Counters', 'Pres'],
        'Hits': [team_data['Hits_Singles'].sum(), team_data['Hits_Multi'].sum(), team_data['Hits_Counters'].sum(), team_data['Hits_Pres'].sum()],
        'Throws': [team_data['Throws_Singles'].sum(), team_data['Throws_Multi'].sum(), team_data['Throws_Counters'].sum(), team_data['Throws_Pres'].sum()],
        'Dodges': [team_data['Dodges_Singles'].sum(), team_data['Dodges_Multi'].sum(), team_data['Dodges_Counters'].sum(), team_data['Dodges_Pres'].sum()],
        'Blocks': [team_data['Blocks_Singles'].sum(), team_data['Blocks_Multi'].sum(), team_data['Blocks_Counters'].sum(), team_data['Blocks_Pres'].sum()]
    }
    situational_df = pd.DataFrame(situational_totals)

    fig = go.Figure()
    fig.add_trace(go.Bar(x=situational_df['Situation'], y=situational_df['Hits'], name='Hits'))
    fig.add_trace(go.Bar(x=situational_df['Situation'], y=situational_df['Throws'], name='Throws'))
    fig.add_trace(go.Bar(x=situational_df['Situation'], y=situational_df['Dodges'], name='Dodges'))
    fig.add_trace(go.Bar(x=situational_df['Situation'], y=situational_df['Blocks'], name='Blocks'))

    fig.update_layout(
        barmode='group',
        title=f'<b>Team Action Totals by Situation for {selected_team}</b>',
        xaxis_title="Situation",
        yaxis_title="Total Count"
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # --- Player Contribution ---
    st.subheader("Player Contribution to Team Performance")
    st.dataframe(
        team_data[['Player_ID', 'Overall_Performance', 'K/D_Ratio', 'Hits', 'Catches', 'Dodges', 'Blocks']]
        .sort_values('Overall_Performance', ascending=False),
        use_container_width=True
    )
