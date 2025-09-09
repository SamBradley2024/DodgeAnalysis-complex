# Dodgeball Analytics Dashboard

An AI-powered web application built with Streamlit for deep analysis of dodgeball match and game data. This dashboard transforms raw stats into actionable insights, player role classifications, and predictive forecasts to help coaches and players make data-driven decisions.

## Key Features ✨

The application is a multi-page dashboard, with each page providing a unique analytical perspective.

### 🏠 Home

  * **Dual Data Sources:** Users can either upload their own data via a **CSV file** or connect directly to a live **Google Sheet**.
  * **Data Caching:** Intelligently caches data and models to ensure fast performance after the initial load.

### 📊 League Overview

  * **At-a-Glance Metrics:** Displays key league-wide statistics like total players, teams, and games analyzed.
  * **Visual Dashboard:** A 4-quadrant chart providing a complete overview of the league:
      * Top 10 performing players.
      * A radar chart comparing the skills of top teams.
      * A pie chart of AI-generated player role distributions.
      * A Performance vs. Consistency scatter plot to categorize player types (Stars, Wildcards, etc.).
  * **Interactive Leaderboards:** A tabbed view to rank players by every major statistic, including Overall Performance, K/D Ratio, Hit Accuracy, and more.

### 👤 Player Analysis

  * **Individual Deep Dive:** Select any player to see a detailed breakdown of their performance.
  * **Comprehensive Dashboard:** Visualizes a player's average skills (radar chart), performance trend across games, and win/loss record.
  * **Stamina Analysis:** A unique line chart that tracks a player's performance across the games *within* matches to analyze their endurance and consistency over time.
  * **Elimination Profile:** Provides coaching insights based on whether a player is more frequently eliminated by being hit or having their throws caught.

### 🏆 Team Analysis

  * **Team-Specific View:** Focus on a single team to analyze its overall performance and player composition.
  * **Strategic Dashboard:** Includes charts on player role distribution within the team, overall game outcomes, and an Offensive vs. Defensive Rating scatter plot to identify player specializations.

### 🎲 Match Analysis

  * **Match-Centric View:** The app understands the hierarchy of Matches (a 15-minute period) and Games (a single round).
  * **Match Selector:** Choose a specific `Match_ID` to analyze.
  * **Dual-View Analysis:**
      * **Match Summary:** A game-by-game breakdown of the match, showing the winner and MVP of each game.
      * **Individual Performance:** A complete leaderboard of every player's stats aggregated across all games within that single match.

### 🤖 AI Insights & Predictor

  * **Automated Strategic Insights:** The AI scans the entire dataset to find:
      * The **Key to Success**: The single most important statistic that correlates with winning performance.
      * Team-specific **Statistical Gaps**: Analyzes every team to find their biggest statistical weakness compared to the league average.
  * **🔮 AI Matchup Predictor:** An interactive tool to forecast game outcomes.
      * Select any two teams from the dataset.
      * The AI uses a trained Random Forest model to predict the win probability for each team based on their historical performance data.

### 🧑‍🏫 Coaching Corner

  * **Actionable Advice:** Provides AI-generated coaching recommendations for any selected player or team, highlighting their specific areas for improvement.

-----

## Data Requirements

To use the app, your data source (CSV or Google Sheet) **must** contain the following columns. The `Match_ID` column is essential for the new match-level analysis features.

  * `Match_ID`: Identifier for a 15-minute match (e.g., `Match_1`).
  * `Game_ID`: Unique identifier for a single game/round.
  * `Player_ID`: Unique identifier for each player.
  * `Team`: The team the player belongs to.
  * `Game_Outcome`: Result for the player's team in that game ('Win' or 'Loss').
  * `Hits`: Number of opponents the player hit.
  * `Throws`: Total number of throws.
  * `Catches`: Number of opponent throws the player caught.
  * `Dodges`: Number of times the player successfully dodged a throw.
  * `Blocks`: Number of times the player blocked a throw.
  * `Hit_Out`: Number of times the player was eliminated by being hit.
  * `Caught_Out`: Number of times the player was eliminated because their throw was caught.

-----

## 🚀 Getting Started

### Prerequisites

  * Python 3.9+
  * A Google Cloud Platform project with the Google Sheets API and Google Drive API enabled.

### Installation & Setup

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/SamBradley2024/DodgeAnalysis.git
    cd DodgeAnalysis
    ```

2.  **Create and activate a virtual environment:**

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

4.  **Set up Streamlit Secrets for Google Sheets:**

      * Follow the Streamlit documentation to create a `secrets.toml` file in a `.streamlit` directory.
      * Add your Google Service Account credentials to this file in the format provided by Streamlit.

5.  **Run the application:**

    ```bash
    streamlit run Home.py
    ```

    Your app will now be running in your browser.

-----

## 🛠️ Technologies Used

  * **Framework:** Streamlit
  * **Data Manipulation:** Pandas
  * **Data Visualization:** Plotly
  * **Machine Learning:** Scikit-learn
  * **Google Sheets Integration:** Gspread
