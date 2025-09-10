# **Dodgeball Analytics Dashboard**

An AI-powered web application built with Streamlit for deep analysis of dodgeball match data. This dashboard transforms raw, pivoted stats from multiple games into actionable insights, player role classifications, and advanced visualizations to help coaches and players make data-driven decisions.

## **Key Features ✨**

The application is a multi-page dashboard, with each page providing a unique analytical perspective based on all loaded game data.

### **🏠 Home**

* **Multi-Game Data Loading:** Users can either upload **multiple CSV files** at once or connect to a live **Google Sheet** and select **multiple worksheets**.  
* **Game-Centric Logic:** Each selected sheet or file is treated as a unique game, allowing the application to build a comprehensive career dataset for trend and average analysis.  
* **Intelligent Caching:** Caches the processed data to ensure fast performance and responsiveness after the initial load.

### **🏆 League Overview**

* **At-a-Glance Metrics:** Displays key league-wide statistics like total players, teams, and games analyzed.  
* **Visual Dashboard:** A 2x2 chart providing a complete overview of the league's strategic landscape:  
  * An interactive bar chart of the top 10 performing players for any selected metric.  
  * A radar chart comparing the aggregate skills of the top teams.  
  * A pie chart showing the distribution of AI-generated player roles across the league.  
  * An **Offense vs. Defense** scatter plot to categorize player styles and impact.  
* **Dynamic Leaderboards:** A tabbed interface to view leaderboards for every major statistic, from Overall\_Performance to specific situational stats.

### **👤 Player Analysis**

* **Dual Analysis Mode:** A powerful dashboard that can be toggled to analyze a player's performance based on:  
  * **Career Averages:** Aggregates stats from all loaded games to show a player's typical performance.  
  * **Single Game Analysis:** Isolates a specific game to analyze a player's performance in that context.  
* **Advanced Situational Charts:**  
  * **Offensive Breakdown:** A stacked bar chart showing **Hits vs. Misses** for different scenarios (Singles, Multi-ball, etc.), with a clear annotation for **Hit Accuracy %**.  
  * **Defensive Breakdown:** A stacked bar chart showing **Dodges, Blocks, vs. being Hit Out**, with a clear annotation for **Survivability %**.  
* **Game-by-Game Trend Chart:** An interactive line chart to track any single statistic over time, making it easy to compare performance between games (sheets).

### **drużyna Team Analysis**

* **Team-Specific Deep Dive:** Focus on a single team to analyze its overall performance, player composition, and key statistics aggregated across all loaded games.  
* **Strategic Dashboard:** Includes charts on player role distribution within the team, overall game outcomes, and an Offensive vs. Defensive Rating scatter plot to identify player specializations.

### **🎲 Game Analysis**

* **Game-Centric View:** Select any single game (sheet/file) from the loaded data to perform a deep dive.  
* **Transposed Data Table:** Displays a detailed statistical breakdown with **players as columns** and **stats as rows** for easy side-by-side comparison.  
* **Simple vs. In-depth View:** A toggle to switch between a curated table of key stats and a comprehensive table of every metric available in the source file.  
* **Performance vs. Career Average:** A bar chart that instantly highlights which players over- or under-performed in that specific game compared to their career average across all other games.

### **🤖 AI Insights & Advanced Analytics**

* **Automated Strategic Insights:** The AI scans the entire dataset to find:  
  * **Standout Situational Performers**: Identifies players who excel in specific scenarios (e.g., best counter-attacker, most survivable in multi-ball).  
  * **Team-level Tactical Patterns**: Analyzes team strengths and weaknesses in different situations.  
* **Player Specialization Heatmap:** A heatmap that identifies player archetypes by comparing their situational stats to the league average, highlighting standout specialists.

### **👨‍🏫 Coaching Corner**

* **Advanced Player Reports:** Generates a detailed coaching report for any player, identifying their **top 3 strengths and weaknesses** by comparing their stats to both the league average and the average for their specific, AI-generated player role.  
* **Dual Comparison Charts:** Provides two clear, large bar charts comparing the player against the league and their role, offering deep, contextual coaching advice.

## **Data Requirements**

The application is built to parse a specific **pivoted data format**. For successful data loading, your source files (CSV or Google Sheet) **must** adhere to the following structure:

* The data must be in a table where **metrics are the rows** (e.g., Overall Hits, Dodges (singles)).  
* **Players must be the columns.**  
* The very first cell (A1) should contain the team name (e.g., vs Silverbacks 2s).  
* The application will automatically handle and convert all other data points.

#### **Example Snippet:**

vs Team Name,Player A,Player B,Player C  
Overall Hits,10,5,12  
Overall Throws,20,15,18  
Catches made,2,0,3  
Dodges (overall),5,8,2  
...

## **🚀 Getting Started**

### **Prerequisites**

* Python 3.9+  
* A Google Cloud Platform project with the Google Sheets API and Google Drive API enabled.

### **Installation & Setup**

1. **Clone the repository:**  
   git clone \<your-repository-url\>  
   cd \<your-repository-name\>

2. **Create and activate a virtual environment:**  
   python \-m venv venv  
   source venv/bin/activate  \# On Windows, use \`venv\\Scripts\\activate\`

3. Install dependencies:  
   Create a requirements.txt file with the following content:  
   streamlit  
   pandas  
   plotly  
   scikit-learn  
   gspread  
   google-auth-oauthlib

   Then run the installation command:  
   pip install \-r requirements.txt

4. **Set up Streamlit Secrets for Google Sheets:**  
   * Follow the [Streamlit documentation](https://www.google.com/search?q=https://docs.streamlit.io/knowledge-base/tutorials/databases/g-sheets) to create a secrets.toml file in a .streamlit directory.  
   * Add your Google Service Account credentials to this file in the format specified by Streamlit.  
5. **Run the application:**  
   streamlit run Home.py

   Your app will now be running in your browser.

## **🛠️ Technologies Used**

* **Framework:** Streamlit  
* **Data Manipulation:** Pandas  
* **Data Visualization:** Plotly  
* **Machine Learning:** Scikit-learn  
* **Google Sheets Integration:** Gspread