import streamlit as st
import pandas as pd
import plotly.express as px
import utils

# --- Page Configuration and State Check ---
st.set_page_config(page_title="Advanced Analytics", page_icon="🔬", layout="wide")
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
st.header("🔬 Advanced Analytics")
st.info(f"Analyzing detailed situational data from: **{st.session_state.source_name}**")

tab1, tab2, tab3 = st.tabs(["Situational Specialization", "Key Performance Correlators", "Generated Strategic & Situational Insights"])

with tab1:
    st.subheader(
        "Player Situational Specialization",
        help="This analysis identifies players with standout skills in specific situations compared to the league average."
    )
    
    spec_stats = [
        'Hits_Singles', 'Hits_Multi', 'Hits_Counters',
        'Dodges_Singles', 'Dodges_Multi', 'Dodges_Counters',
        'Blocks_Singles', 'Blocks_Multi', 'Blocks_Counters',
        'Survivability_Singles', 'Survivability_Multi', 'Survivability_Counters'
    ]
    existing_spec_stats = [col for col in spec_stats if col in df.columns]
    
    if not existing_spec_stats:
        st.warning("No situational data available for specialization analysis.")
    else:
        player_avg_stats = df.groupby('Player_ID')[existing_spec_stats].mean()
        league_avg = player_avg_stats.mean()
        specialization = player_avg_stats / (league_avg + 1e-6) # Add epsilon to avoid division by zero
        
        # Select top 20 players with the most variance in their specialization
        top_specialized_players = specialization.std(axis=1).nlargest(20).index
        specialization_subset = specialization.loc[top_specialized_players]
        
        fig = px.imshow(
            specialization_subset,
            text_auto=".2f",
            aspect="auto",
            color_continuous_scale='Viridis',
            labels=dict(x="Situation", y="Player", color="Specialization (x League Avg)"),
            title="<b>Player Specialization Heatmap (vs. League Average)</b>"
        )
        st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.subheader(
        "Key Performance Correlators",
        help="This analysis identifies which specific situational stats have the strongest relationship with a player's overall performance score."
    )
    
    corr_cols = [
        'Hits_Singles', 'Hits_Multi', 'Hits_Counters',
        'Dodges_Singles', 'Dodges_Multi', 'Dodges_Counters',
        'Blocks_Singles', 'Blocks_Multi', 'Blocks_Counters',
        'Survivability_Singles', 'Survivability_Multi', 'Survivability_Counters',
        'Catches_Attempted', 'K/D_Ratio', 'Overall_Performance'
    ]
    
    existing_corr_cols = [col for col in corr_cols if col in df.columns]

    if 'Overall_Performance' in existing_corr_cols and len(existing_corr_cols) > 1:
        performance_corr = df[existing_corr_cols].corr()['Overall_Performance'].abs().sort_values(ascending=False).drop('Overall_Performance').head(10)
        
        if not performance_corr.empty:
            corr_df = pd.DataFrame({'Metric': performance_corr.index, 'Correlation': performance_corr.values})
            fig = px.bar(
                corr_df, x='Correlation', y='Metric',
                title='<b>Top 10 Situational Metrics Correlated with Performance</b>',
                color='Correlation', color_continuous_scale='plasma', text_auto='.2f'
            )
            fig.update_layout(yaxis={'categoryorder': 'total ascending'})
            st.plotly_chart(fig, use_container_width=True)
            st.info("💡 A higher correlation suggests that excelling in this specific situation is a strong indicator of a high overall performance score.")
        else:
            st.warning("Could not calculate correlations.")
    else:
        st.warning("Not enough situational data or the 'Overall_Performance' column is missing.")

with tab3:
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
