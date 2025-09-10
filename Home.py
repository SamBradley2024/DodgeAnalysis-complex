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

# Define the two options in columns
col1, col2 = st.columns(2)

# --- Option 1: Google Sheets ---
with col1:
    st.subheader("🔗 Option 1: Use Live Google Sheet")
    st.write("Connect to the 'Dodgeball App Data' Google Sheet. Any edits you make to the sheet will be reflected in the app.")
    
    sheet_names = utils.get_worksheet_names()
    if sheet_names:
        selected_sheet = st.selectbox("Select a worksheet", sheet_names)

        if st.button("Load from Google Sheet"):
            # ADDED: Spinner provides visual feedback that the app is working.
            with st.spinner(f"Loading and processing '{selected_sheet}'..."):
                # Call the function to get the data
                raw_df = utils.load_and_process_google_sheet(selected_sheet)
                
                # MODIFIED: A more robust check for a valid, non-empty DataFrame.
                if raw_df is not None and not raw_df.empty:
                    st.success("✅ Data loaded successfully!")
                    
                    st.cache_data.clear()
                    st.cache_resource.clear()
                    
                    utils.initialize_app(raw_df, f"Google Sheet: {selected_sheet}")
                    st.rerun()
                else:
                    # This error message is now guaranteed to show if loading fails.
                    st.error("❌ Failed to process data from the selected sheet. This might be due to a connection issue or an unexpected data format.")
    else:
        st.warning("Could not retrieve worksheet names. Check Google Sheets connection and secrets configuration.")

# --- Option 2: CSV Upload ---
with col2:
    st.subheader("📄 Option 2: Upload a CSV File")
    st.write("Upload your own dodgeball data. The file must have the same column headers as the template.")
    
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

    if uploaded_file is not None:
        # Using a spinner here as well for consistency
        with st.spinner("Processing CSV file..."):
            raw_df = utils.load_and_process_custom_csv(uploaded_file)
            
            if raw_df is not None and not raw_df.empty:
                st.success("✅ CSV processed successfully!")

                st.cache_data.clear()
                st.cache_resource.clear()

                utils.initialize_app(raw_df, f"Uploaded File: {uploaded_file.name}")
                st.rerun()
            else:
                st.error("❌ Failed to process the uploaded CSV file.")

