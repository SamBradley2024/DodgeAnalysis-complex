import streamlit as st
import pandas as pd
import utils

# --- Page Configuration ---
st.set_page_config(
    page_title="Dodgeball Analytics - Welcome",
    page_icon="🤾",
    layout="wide"
)

st.markdown(utils.load_css(), unsafe_allow_html=True)

# --- State Check ---
if 'data_loaded' in st.session_state and st.session_state.data_loaded:
    st.success(f"Data from **{st.session_state.source_name}** is already loaded.")
    st.info("Navigate to any page on the left to start the analysis. To use a different data source, please select another from this page.")

# --- Welcome and Data Source Selection ---
st.markdown("""
<div class="main-header">
    <h1>Welcome to the Dodgeball Analytics Dashboard</h1>
    <p>Please choose a data source to begin your analysis.</p>
</div>
""", unsafe_allow_html=True)

col1, col2 = st.columns(2)

# --- Option 1: Google Sheets ---
with col1:
    st.subheader("🔗 Option 1: Use Live Google Sheet")
    st.write("Connect to the 'Dodgeball App Data' Google Sheet.")
    
    # Get worksheet names and create the selection box
    try:
        sheet_names = utils.get_worksheet_names()
        selected_sheet = st.selectbox("Select a worksheet", sheet_names)

        # Re-enabled the button and added the cache clearing logic
        if st.button("Load from Google Sheet"):
            raw_df = utils.load_from_google_sheet(selected_sheet)
            if raw_df is not None:
                # These two lines clear the cache to ensure new data is loaded
                st.cache_data.clear()
                st.cache_resource.clear()
                
                utils.initialize_app(raw_df, f"Google Sheet: {selected_sheet}")
                st.rerun()
    except Exception as e:
        st.error("Could not connect to Google Sheets. Check credentials.")


# --- Option 2: CSV Upload (MODIFIED) ---
with col2:
    st.subheader("📄 Option 2: Upload CSV Stat Files")
    st.write("Upload one or more of your 'Meet' data files.")
    
    uploaded_files = st.file_uploader(
        "Choose CSV file(s)", 
        type="csv",
        accept_multiple_files=True
    )

    if uploaded_files:
        # The button is now always available when files are staged
        if st.button("Process Uploaded Files"):
            # Call the NEW, corrected processing function from utils.py
            raw_df = utils.load_and_process_uploaded_csvs(uploaded_files)
            
            if not raw_df.empty:
                st.cache_data.clear()
                st.cache_resource.clear()
                
                # Initialize the app with the processed DataFrame
                utils.initialize_app(raw_df, "Uploaded CSV Files")
                st.rerun()

