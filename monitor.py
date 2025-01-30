import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px  # Using Plotly for enhanced visuals
from sklearn.ensemble import IsolationForest
import io  # For handling file downloads

# ------------------------------------------------------------------------------------
# App Configuration
# ------------------------------------------------------------------------------------
st.set_page_config(
    page_title="Smart Water Management System",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ------------------------------------------------------------------------------------
# Title and Description
# ------------------------------------------------------------------------------------
st.title("💧 Smart Water Management System")

st.markdown(
    """
    Welcome to the **Smart Water Management System**! This web app helps you track and optimize your water usage by detecting anomalies and providing detailed tips for sustainable water consumption.

    **Features:**
    - **Data Upload**: Easily upload your water usage data.
    - **Anomaly Detection**: Identify unusual water consumption patterns.
    - **Interactive Visualization**: Explore your data with dynamic plots.
    - **Personalized Tips**: Receive recommendations based on your usage.
    - **Download Results**: Export your analysis for further inspection.
    """
)

# ------------------------------------------------------------------------------------
# Sidebar for User Inputs
# ------------------------------------------------------------------------------------
st.sidebar.header("📥 Upload Your Data")

# File upload
uploaded_file = st.sidebar.file_uploader(
    "Upload your CSV file here",
    type=["csv"],
    help="Ensure your CSV has at least two columns: one for dates and one for water usage."
)

# Initialize default data
default_data = {
    'Date': pd.date_range(start='2025-01-01', periods=7, freq='D'),
    'Water Usage': [200, 150, 220, 300, 1000, 250, 180]
}

# ------------------------------------------------------------------------------------
# Handle User Data (Uploaded File or Manual Input)
# ------------------------------------------------------------------------------------
if uploaded_file is not None:
    try:
        df_uploaded = pd.read_csv(uploaded_file)
        st.sidebar.success("✅ File uploaded successfully!")

        # Let user map their columns to 'Date' and 'Water Usage'
        all_columns = df_uploaded.columns.tolist()
        date_column = st.sidebar.selectbox("Select the Date column", options=all_columns)
        usage_column = st.sidebar.selectbox("Select the Water Usage column", options=all_columns)

        # Convert to datetime
        df_uploaded[date_column] = pd.to_datetime(df_uploaded[date_column], errors='coerce')
        if df_uploaded[date_column].isnull().any():
            st.sidebar.warning("⚠️ Some dates couldn't be parsed. Please check your date format.")
        
        # Drop rows with invalid dates or usage
        df_uploaded = df_uploaded.dropna(subset=[date_column, usage_column])

        # Assign to standardized names
        df = pd.DataFrame({
            'Date': df_uploaded[date_column],
            'Water Usage': df_uploaded[usage_column]
        })

    except Exception as e:
        st.sidebar.error(f"❌ Error loading file: {e}")
        st.warning("Using default data instead.")
        df = pd.DataFrame(default_data)
else:
    st.sidebar.info("📄 No file uploaded. Using default data.")
    df = pd.DataFrame(default_data)

# ------------------------------------------------------------------------------------
# Data Preprocessing
# ------------------------------------------------------------------------------------
df.sort_values('Date', inplace=True)
df.reset_index(drop=True, inplace=True)

# **Ensure 'Anomaly' column exists before visualization**
df["Anomaly"] = "Normal"  # Default values

# ------------------------------------------------------------------------------------
# Layout: Two Columns for Controls and Visualization
# ------------------------------------------------------------------------------------
col1, col2 = st.columns([1, 2])

# ------------------------------------------------------------------------------------
# Column 1: Anomaly Detection Controls
# ------------------------------------------------------------------------------------
with col1:
    st.header("⚙️ Anomaly Detection Settings")
    contamination = st.slider(
        "Anomaly Detection Sensitivity",
        min_value=0.01,
        max_value=0.5,
        value=0.2,
        step=0.01,
        help="Higher contamination values will flag more data points as anomalies."
    )

    # Button to trigger anomaly detection
    detect_anomalies = st.button("🔍 Detect Anomalies")

# ------------------------------------------------------------------------------------
# Column 2: Interactive Plot
# ------------------------------------------------------------------------------------
with col2:
    st.header("📈 Water Usage Visualization")

    if detect_anomalies:
        # Isolation Forest for anomaly detection
        model = IsolationForest(contamination=contamination, random_state=42)
        df['Anomaly'] = model.fit_predict(df[['Water Usage']])
        df['Anomaly'] = df['Anomaly'].map({1: 'Normal', -1: 'Anomaly'})

        # Highlight anomalies
        anomalies = df[df['Anomaly'] == 'Anomaly']

        # Interactive Plotly Line Chart
        fig = px.line(
            df, 
            x='Date', 
            y='Water Usage',
            title='Water Usage Over Time with Anomalies',
            labels={'Water Usage': 'Water Usage (Liters)', 'Date': 'Date'},
            markers=True
        )

        # Add anomalies as scatter points
        fig.add_scatter(
            x=anomalies['Date'],
            y=anomalies['Water Usage'],
            mode='markers',
            marker=dict(color='red', size=10, symbol='x'),
            name='Anomalies'
        )

        fig.update_layout(template='plotly_white', hovermode='x unified')
        st.plotly_chart(fig, use_container_width=True)
    else:
        fig = px.line(
            df, 
            x='Date', 
            y='Water Usage',
            title='Water Usage Over Time',
            labels={'Water Usage': 'Water Usage (Liters)', 'Date': 'Date'},
            markers=True
        )
        fig.update_layout(template='plotly_white', hovermode='x unified')
        st.plotly_chart(fig, use_container_width=True)

# ------------------------------------------------------------------------------------
# Display Data with Anomalies
# ------------------------------------------------------------------------------------
st.subheader("📊 Water Usage Data")
st.dataframe(df.style.apply(
    lambda row: ['background-color: #FFCCCC' if row['Anomaly'] == 'Anomaly' else '' for _ in row],
    axis=1
))

# ------------------------------------------------------------------------------------
# Download Analysis Results
# ------------------------------------------------------------------------------------
st.subheader("💾 Download Your Analysis")
csv_buffer = io.StringIO()
df.to_csv(csv_buffer, index=False)
csv_data = csv_buffer.getvalue()

st.download_button(
    label="📥 Download CSV",
    data=csv_data,
    file_name="water_usage_analysis.csv",
    mime="text/csv"
)


# ------------------------------------------------------------------------------------
# Conservation Tips for Anomalous Usage
# ------------------------------------------------------------------------------------
st.subheader("🔍 Conservation Tips for Detected Anomalies")

if df['Anomaly'].str.contains('Anomaly').any():
    for _, row in df[df['Anomaly'] == 'Anomaly'].iterrows():
        with st.expander(f"💡 Tips for {row['Date'].date()}"):
            st.markdown(
                f"### ⚠️ High Water Usage Detected on **{row['Date'].date()}**"
            )
            st.markdown(
                """
                **Recommendations:**
                - Fix leaks.
                - Use water-saving fixtures.
                - Optimize appliance use.
                - Shorten showers.
                - Use drought-resistant plants.
                """
            )
else:
    st.info("🎉 No anomalies detected! Keep up the good water management practices.")

# ------------------------------------------------------------------------------------
# General Water-Saving Tips
# ------------------------------------------------------------------------------------
st.subheader("🌍 General Water Conservation Recommendations")

if st.button('💡 Show General Water Conservation Tips'):
    st.markdown(
        """
        - Use water-efficient appliances.
        - Fix leaks promptly.
        - Water plants during non-peak hours.
        - Shorten showers.
        - Collect rainwater.
        - Educate your household.
        """
    )
