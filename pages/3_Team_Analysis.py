import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import utils

# --- Page Configuration and State Check ---
st.set_page_config(page_title="Team Analysis", page_icon="🏆", layout="wide")
st.markdown(utils.load_css(), unsafe_allow_html=True)

if 'data_loaded' not in st.session_state or not st.session_state.data_loaded:
    st.warning("Please select and load a data source from the 🏠 Home page first.")
    st.stop()

df = st.session_state.df_enhanced

# --- Team Selection ---
st.header("🏆 Team Analysis")
team_list = sorted(df['Team'].unique())
if not team_list:
    st.warning("No teams found in the selected data source.")
    st.stop()
    
selected_team = st.selectbox("Select Team", team_list)

if selected_team:
    # Filter for the selected team across all games
    team_all_games = df[df['Team'] == selected_team].copy()
    
    st.subheader(f"Overall Performance for {selected_team}")
    
    # --- Career Summary Metrics ---
    team_career_summary = team_all_games.mean(numeric_only=True)
    player_count = team_all_games['Player_ID'].nunique()
    game_count = team_all_games['Game_ID'].nunique()

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        utils.styled_metric("Total Players", f"{player_count}")
    with col2:
        utils.styled_metric("Games Played", f"{game_count}")
    with col3:
        utils.styled_metric("Avg Team Performance", f"{team_career_summary.get('Overall_Performance', 0):.2f}")
    with col4:
        utils.styled_metric("Avg Team K/D Ratio", f"{team_career_summary.get('K/D_Ratio', 0):.2f}")
    
    st.markdown("---")

    # --- Analysis Tabs ---
    tab1, tab2, tab3 = st.tabs(["Player Roster", "Team Situational Stats", "Role Distribution"])

    with tab1:
        st.subheader("Player Roster and Key Stats (Career Averages)")
        player_summary = team_all_games.groupby('Player_ID').agg(
            Overall_Performance=('Overall_Performance', 'mean'),
            K_D_Ratio=('K/D_Ratio', 'mean'),
            Hits=('Hits', 'mean'),
            Catches=('Catches', 'mean'),
            Dodges=('Dodges', 'mean'),
            Player_Role=('Player_Role', 'first')
        ).reset_index().sort_values('Overall_Performance', ascending=False)
        
        st.dataframe(player_summary.style.format({
            'Overall_Performance': '{:.2f}', 'K_D_Ratio': '{:.2f}', 'Hits': '{:.2f}', 'Catches': '{:.2f}', 'Dodges': '{:.2f}'
        }), use_container_width=True)

    with tab2:
        st.subheader("Team Situational Performance (Averages per Game)")
        
        # Using the team's career (cross-game) averages for the chart
        fig = go.Figure()
        fig.add_trace(go.Bar(name='Avg Hits', x=['Singles', 'Multi-ball', 'Counters'], y=[team_career_summary.get('Hits_Singles', 0), team_career_summary.get('Hits_Multi', 0), team_career_summary.get('Hits_Counters', 0)]))
        fig.add_trace(go.Bar(name='Avg Throws', x=['Singles', 'Multi-ball', 'Counters'], y=[team_career_summary.get('Throws_Singles', 0), team_career_summary.get('Throws_Multi', 0), team_career_summary.get('Throws_Counters', 0)]))
        fig.add_trace(go.Bar(name='Avg Dodges', x=['Singles', 'Multi-ball', 'Counters'], y=[team_career_summary.get('Dodges_Singles', 0), team_career_summary.get('Dodges_Multi', 0), team_career_summary.get('Dodges_Counters', 0)]))
        fig.update_layout(barmode='group', title_text="<b>Team's Situational Strengths (Career Averages)</b>")
        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        st.subheader("Player Role Distribution")
        if 'Player_Role' in team_all_games.columns and team_all_games['Player_Role'].nunique() > 1:
            role_counts = team_all_games.drop_duplicates(subset=['Player_ID'])['Player_Role'].value_counts()
            fig2 = px.pie(
                values=role_counts.values,
                names=role_counts.index,
                title="Team Composition by Player Role"
            )
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.info("Role analysis requires more data or player variation.")

