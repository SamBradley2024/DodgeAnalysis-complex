import streamlit as st
import pandas as pd
import utils 

# --- State Management and Sidebar ---
st.set_page_config(page_title="AI Insights", page_icon="🤖", layout="wide")
st.markdown(utils.load_css(), unsafe_allow_html=True)

# Check if data has been loaded from the Home page
if 'data_loaded' not in st.session_state or not st.session_state.data_loaded:
    st.warning("Please select and load a data source from the 🏠 Home page first.")
    st.stop()

# If data is loaded, get it from session state
df = st.session_state.df_enhanced
models = st.session_state.models

# --- Page Content ---
st.header("AI Insights")
st.info(f"This section provides automated insights and predictive models based on the **{st.session_state.source_name}** dataset.")
st.markdown("---")

# --- Section 1: Generated Insights ---
st.subheader(
    "Generated Strategic Insights",
    help="This section displays automated insights discovered by analyzing the entire dataset. It identifies top performers, the most critical skills for success, and potential strategic gaps."
)

insights = utils.generate_insights(df, models)

if not insights:
    st.warning("Could not generate any advanced insights. The dataset may be too small or lack sufficient variation for the AI models to find patterns.")
else:
    for insight in insights:
        st.markdown(f"""
        <div class.insight-box">
            <p>{insight}</p>
        </div>
        """, unsafe_allow_html=True)

st.markdown("---")

# --- Section 2: Interactive Win Predictor ---
st.subheader(
    "Matchup Predictor",
    help="This interactive tool uses a machine learning model to forecast the outcome of a game between two teams. The prediction is based on the historical average performance, K/D ratio, and hit accuracy of each team's players."
)

# Check if the win predictor model was successfully trained
if 'win_predictor' not in models:
    st.error("The Win Prediction Model could not be trained due to insufficient historical game data (requires at least 10 games with two teams).")
    st.stop()

# --- Team Selection ---
team_list = sorted(df['Team'].unique())
if len(team_list) < 2:
    st.warning("You need at least two teams in the dataset to run a prediction.")
    st.stop()

col1, col2 = st.columns(2)
with col1:
    team_a = st.selectbox("Select Team A", team_list, index=0)
with col2:
    team_b_options = [t for t in team_list if t != team_a]
    team_b = st.selectbox("Select Team B", team_b_options, index=0 if not team_b_options else min(1, len(team_b_options)-1))

# --- Prediction Logic ---
if st.button("Predict Outcome"):
    if team_a and team_b:
        win_model, win_model_features = models['win_predictor']
        
        team_stats_summary = df.groupby('Team').agg(
            Avg_Performance=('Avg_Performance', 'first'),
            Avg_KD_Ratio=('Avg_KD_Ratio', 'first'),
            Avg_Hit_Accuracy=('Avg_Hit_Accuracy', 'first')
        ).fillna(0) # Fill NA for teams that might be missing aggregated stats
        
        team_a_hist_stats = team_stats_summary.loc[team_a]
        team_b_hist_stats = team_stats_summary.loc[team_b]
        
        prediction_input = pd.DataFrame([{
            'Team_A_Avg_Perf': team_a_hist_stats['Avg_Performance'],
            'Team_A_Avg_KD': team_a_hist_stats['Avg_KD_Ratio'],
            'Team_A_Avg_Acc': team_a_hist_stats['Avg_Hit_Accuracy'],
            'Team_B_Avg_Perf': team_b_hist_stats['Avg_Performance'],
            'Team_B_Avg_KD': team_b_hist_stats['Avg_KD_Ratio'],
            'Team_B_Avg_Acc': team_b_hist_stats['Avg_Hit_Accuracy'],
        }])
        
        prediction_input = prediction_input[win_model_features]

        probabilities = win_model.predict_proba(prediction_input)[0]
        prob_team_a_wins = probabilities[1]
        prob_team_b_wins = probabilities[0]

        st.subheader("Prediction Result")
        res_col1, res_col2 = st.columns(2)
        with res_col1:
            st.metric(label=f"**{team_a}** Win Probability", value=f"{prob_team_a_wins:.1%}")
        with res_col2:
            st.metric(label=f"**{team_b}** Win Probability", value=f"{prob_team_b_wins:.1%}")

        if abs(prob_team_a_wins - prob_team_b_wins) < 0.1:
             st.info("⚖️ The model considers this a very close matchup, almost a toss-up.")
        elif prob_team_a_wins > prob_team_b_wins:
            st.success(f"🏆 **{team_a}** is the predicted winner.")
        else:
            st.success(f"🏆 **{team_b}** is the predicted winner.")
        
        st.caption(f"Model accuracy based on historical test data: {st.session_state.get('win_model_accuracy', 0):.1%}")