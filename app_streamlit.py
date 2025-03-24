import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import folium
from streamlit_folium import folium_static
import glob
import datetime
import requests

API_URL = "https://router.huggingface.co/hf-inference/models/mistralai/Mistral-7B-Instruct-v0.3/v1/chat/completions"
API_KEY = st.secrets["huggingface"]["api_key"]
headers = {"Authorization": f"Bearer {API_KEY}"}


# Function to load and merge pollutant data
def load_data():
    pollutant_files = glob.glob("predicted/*.csv")
    data_frames = []
    for file in pollutant_files:
        # Extract pollutant name from the filename, e.g., gurugram_pm_25_predictions_with_forecast.csv
        pollutant = file.split('_')[1].lower()  # Ensure lowercase for consistency
        df = pd.read_csv(file)
        # Rename columns for consistency
        df.rename(columns={
            'element': pollutant,  # Rename 'element' to the pollutant name
            'element_predicted': f'predicted_{pollutant}'  # Rename 'element_predicted'
        }, inplace=True)
        # Standardize the 'location' column
        df['location'] = df['location'].str.strip().str.lower()
        data_frames.append(df)
    final_data = pd.DataFrame()
    for i, df in enumerate(data_frames):
        if final_data.empty:
            final_data = df
        else:
            final_data = pd.merge(
                final_data, df, 
                on=['date', 'location', 'lat', 'long', 'id'],  
                how='outer',
                suffixes=('', '_dup')
            )
        # Handle duplicate columns by keeping only the first occurrence
        for col in final_data.columns:
            if col.endswith('_dup'):
                original_col = col.replace('_dup', '')
                if original_col in final_data.columns:
                    final_data[original_col] = final_data[original_col].combine_first(final_data[col])
                    final_data.drop(columns=[col], inplace=True)
                else:
                    final_data.rename(columns={col: original_col}, inplace=True)
    final_data['date'] = pd.to_datetime(final_data['date'])
    # Restore original case for location (capitalize first letter)
    final_data['location'] = final_data['location'].str.title()
    return final_data

n_data = load_data()

new_data = pd.read_csv('./data_collected_from_field_using_mobile_app.csv')

new_dataset = new_data[['lat', 'long', 'description','category', 'image_filename', 'data_id']]

nv_data = pd.merge(
            n_data, new_dataset, 
            left_on='id', right_on='data_id',
            how='outer',  # Use 'outer' to include all rows from both DataFrames
            suffixes=('', '_dup')  # Add a suffix to duplicate columns
        )

# Handle duplicate columns by keeping only the first occurrence
for col in nv_data.columns:
    if col.endswith('_dup'):
        original_col = col.replace('_dup', '')
        if original_col in nv_data.columns:
            # Combine the original and duplicate columns (e.g., take the non-null value)
            nv_data[original_col] = nv_data[original_col].combine_first(nv_data[col])
            # Drop the duplicate column
            nv_data.drop(columns=[col], inplace=True)
        else:
            # Rename the duplicate column back to the original name
            nv_data.rename(columns={col: original_col}, inplace=True)

data = nv_data.dropna(subset=['location'])

# Function to calculate AQI (Placeholder logic)
def calculate_aqi(row):
    # Calculate AQI as the maximum among key pollutant values (actual values used if present)
    aqi = max(
        row.get('pm2.5', 0) if pd.notnull(row.get('pm2.5')) else 0,
        row.get('pm10', 0) if pd.notnull(row.get('pm10')) else 0,
        row.get('no2', 0) if pd.notnull(row.get('no2')) else 0,
        row.get('co', 0) if pd.notnull(row.get('co')) else 0
    )
    return aqi

# Load Data and calculate AQI

data['AQI'] = data.apply(calculate_aqi, axis=1)

# Load Pollution Categories
category_data = pd.DataFrame({
    'id': [1, 2, 3, 4, 5],
    'category': [
        'Industrial Pollution', 'Waste Burning', 'Vehicle Pollution', 
        'Construction & Demolition Waste', 'Brick Kilns'
    ],
    'image': [
        'images/industrial.png', 'images/waste_burning.png', 'images/vehicle.png', 
        'images/construction.png', 'images/brick_kilns.png'
    ]
})

st.set_page_config(page_title="Air Quality Dashboard", layout="wide")
st.title("Air Quality Prediction Dashboard")

# Sidebar: Location, Pollutant, Pollution Category
location = st.sidebar.selectbox("Select Location", sorted(data['location'].unique()))
pollutant = st.sidebar.selectbox("Select Pollutant", ['co', 'pm_25', 'pm_10', 'no2'])
category_selection = st.sidebar.multiselect(
    "Select Pollution Category", ['All'] + category_data['category'].tolist(), default=['All']
)

# Filter Data based on location only (date filter removed)
data_filtered = data[data['location'] == location].sort_values(by='date')

# Display Key Metrics (Temperature, Humidity, AQI) using full data for the location
col1, col2, col3 = st.columns(3)
with col1:
    st.metric(label="Temperature (°C)", value=f"{data_filtered['temp'].mean():.1f}")
with col2:
    st.metric(label="Humidity (%)", value=f"{data_filtered['rh'].mean():.1f}")
with col3:
    st.metric(label="AQI", value=f"{data_filtered['AQI'].mean():.1f}")

# Pollutant Trends: Show actual and predicted values if available
st.markdown(f"### {pollutant.upper()} Trends Over Time")
fig = go.Figure()
fig.add_trace(
    go.Scatter(
        x=data_filtered['date'],
        y=data_filtered[pollutant],
        mode='lines+markers',
        name='Actual'
    )
)
# Check if predicted column exists and add as another trace
predicted_col = f'predicted_{pollutant}'
if predicted_col in data_filtered.columns:
    fig.add_trace(
        go.Scatter(
            x=data_filtered['date'],
            y=data_filtered[predicted_col],
            mode='lines+markers',
            name='Predicted'
        )
    )
fig.update_layout(title=f"{pollutant.upper()} Levels Over Time", xaxis_title="Date", yaxis_title=pollutant.upper())
st.plotly_chart(fig, use_container_width=True)

# Geospatial Heatmap for pollutant trends (using full data for the location)
st.markdown("### Geospatial Pollution Heatmap")
if not data_filtered.empty:
    map_center = [data_filtered.iloc[5]['lat'], data_filtered.iloc[5]['long']]
else:
    map_center = [28.6139, 77.2090]  # Default center if no data is available
map_obj = folium.Map(location=map_center, zoom_start=12)
for _, row in data_filtered.iterrows():
    value_to_plot = row[pollutant] if pd.notna(row[pollutant]) else row.get(predicted_col, None)
    if pd.notna(value_to_plot):
        folium.CircleMarker(
            location=[row['lat'], row['long']],
            radius=max(5, value_to_plot / 10),  # Adjust radius scaling as needed
            color='red',
            fill=True,
            fill_opacity=0.6,
            popup=f"{pollutant.upper()}: {value_to_plot:.2f}"
        ).add_to(map_obj)
folium_static(map_obj, width=0, height=600)

def query(payload):
    response = requests.post(API_URL, headers=headers, json=payload)
    return response.json(), response.status_code

response, statusCode = query({
    "messages": [
        {
            "role": "user",
            "content": f"""
                        Given the air quality index (AQI) of {data_filtered['AQI'].mean()} and the weather conditions described as "{data_filtered['temp'].mean():.1f}" which is in celcius,, generate a short two-sentence advisory:
                        1. Describe the air quality situation concisely.
                        2. Provide a simple recommendation on outdoor activity.

                        Keep it clear and direct, without unnecessary details.
                        """
        }
    ],
    "max_tokens": 50,
    "model": "mistralai/Mistral-7B-Instruct-v0.3"
})

# # Summary Cards
st.markdown("### Air Quality Insights")
col4, col5 = st.columns(2)
with col4:
    if statusCode == 200:
        st.info(response["choices"][0]["message"]['content'].split('\n')[0][3:])
    else:
        if data_filtered['AQI'].mean() < 50:
            st.success("Good time for a morning walk. Moderate pollution levels.")
        else:
            st.info("The air quality is unhealthy for sensitive groups.")
with col5:
    if statusCode == 200:
        st.info(response["choices"][0]["message"]['content'].split('\n')[1][3:])
    else:
        if data_filtered['AQI'].mean() < 50:
            st.success("Good time for a morning walk. Moderate pollution levels.")
        else:
            st.info("Avoid outdoor activities. Unhealthy air quality.")

# Display Pollution Category Images
st.markdown("### Pollution Categories")
if 'All' in category_selection:
    filtered_category_data = data_filtered.dropna(subset=['category'])
else:
    filtered_category_data = data_filtered[data_filtered['category'].isin(category_selection)]

# Get unique categories from the filtered data
unique_categories = filtered_category_data['category'].dropna().unique()

# Display one image per category (if available in the 'image_filename' column)
for cat in unique_categories:
    cat_data = filtered_category_data[filtered_category_data['category'] == cat]
    if 'image_filename' in cat_data.columns and not cat_data['image_filename'].isnull().all():
        image_path = cat_data['image_filename'].iloc[0]  # Use the first image for the category
        st.image(image_path, caption=cat, use_column_width=True)

st.write("### Future Enhancements")
st.write("- Integrate real-time API data\n- Add ML-based predictions\n- Display public health alerts")
