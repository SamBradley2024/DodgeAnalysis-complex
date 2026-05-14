import streamlit as st
import pandas as pd
import utils

st.set_page_config(
    page_title="Dodgeball Analytics",
    layout="wide"
)

st.markdown(utils.load_css(), unsafe_allow_html=True)

if 'data_loaded' in st.session_state and st.session_state.data_loaded:
    st.success(f"Data from **{st.session_state.source_name}** is already loaded.")
    st.info("Use the sidebar to open an analysis page. To switch data sources, load a new one here.")

st.markdown("""
<div class="main-header">
    <h1>Dodgeball Analytics</h1>
    <p>Choose a data source to begin.</p>
</div>
""", unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    st.subheader("Google Sheets")
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
                raw_df = utils.load_and_process_multiple_sheets(selected_sheets)
                if raw_df is not None and not raw_df.empty:
                    utils.initialize_app(raw_df, f"{len(selected_sheets)} Google Sheet(s)")
                    st.rerun()

with col2:
    st.subheader("CSV Upload")
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
            raw_df = utils.load_and_process_multiple_csvs(uploaded_files)
            if raw_df is not None and not raw_df.empty:
                utils.initialize_app(raw_df, f"{len(uploaded_files)} CSV File(s)")
                st.rerun()

