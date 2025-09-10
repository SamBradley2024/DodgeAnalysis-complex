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

# --- State Check: If data is loaded, show a different message ---
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

# --- Option 1: Google Sheets (UPGRADED for multi-select) ---
with col1:
    st.subheader("🔗 Option 1: Use Live Google Sheet(s)")
    st.write("Connect to the 'Dodgeball App Data' Google Sheet and select one or more sheets to load as individual games.")
    
    sheet_names = utils.get_worksheet_names()
    if sheet_names:
        selected_sheets = st.multiselect("Select one or more worksheets (games):", sheet_names)

        if st.button("Load Selected Game(s) from Google Sheets"):
            if not selected_sheets:
                st.warning("Please select at least one worksheet.")
            else:
                st.cache_data.clear()
                st.cache_resource.clear()
                # This new function handles loading and combining multiple sheets
                raw_df = utils.load_and_process_multiple_sheets(selected_sheets)
                if raw_df is not None and not raw_df.empty:
                    utils.initialize_app(raw_df, f"{len(selected_sheets)} Google Sheet(s)")
                    st.rerun()

# --- Option 2: CSV Upload (UPGRADED for multi-upload) ---
with col2:
    st.subheader("📄 Option 2: Upload CSV File(s)")
    st.write("Upload one or more CSV files. Each file will be treated as a separate game.")
    
    uploaded_files = st.file_uploader(
        "Choose one or more CSV files",
        type="csv",
        accept_multiple_files=True
    )

    if uploaded_files:
        if st.button("Load Selected CSV File(s)"):
            st.cache_data.clear()
            st.cache_resource.clear()
            # This new function handles loading and combining multiple CSVs
            raw_df = utils.load_and_process_multiple_csvs(uploaded_files)
            if raw_df is not None and not raw_df.empty:
                utils.initialize_app(raw_df, f"{len(uploaded_files)} CSV File(s)")
                st.rerun()

