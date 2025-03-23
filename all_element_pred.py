import warnings
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# from pandas 
import datetime
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error

def date_parser(x):
    return datetime.datetime.strptime(x, '%Y-%m-%d')

print('Please wait. Importing data...')
multi_df = []

# df = pd.read_csv("vayu_Patna_dynamic_sensor_data_March_2025.csv", encoding = "ISO-8859-1")


# Folder containing the CSV files
folder_path = "./Dataset/Gurugram/Dynamic/"  # Replace with the path to your folder

# Loop through all files in the folder
for file in os.listdir(folder_path):
    if file.endswith(".csv"):  # Check if the file is a CSV
        file_path = os.path.join(folder_path, file)
        # print(f"Reading {file_path}...")
        df1 = pd.read_csv(file_path, encoding="ISO-8859-1")  # Adjust encoding if needed
        multi_df.append(df1)

# combine all DataFrames into one
df = pd.concat(multi_df, ignore_index=True)
for element in ['co', 'no2', 'co2', 'pm_25', 'pm_10','temp', 'rh']:
    print(f"Processing {element} data...")
    # Prepare the data
    gurugram_data_co = df[['data_created_time', f'{element}', 'lat', 'long', 'id']]
    gurugram_data_co = gurugram_data_co.fillna(gurugram_data_co.bfill())


    # Prepare the data
    gurugram_data_co[f'{element}'] = gurugram_data_co[f'{element}'].map(lambda x: str(x))
    gurugram_data_co = gurugram_data_co[gurugram_data_co[f'{element}'] != 'nan']
    gurugram_data_co[f'{element}'] = pd.to_numeric(gurugram_data_co[f'{element}'])
    gurugram_data_co['date'] = gurugram_data_co['data_created_time'].map(lambda x: str(x)[:10])
    gurugram_data_co['date'] = gurugram_data_co['date'].map(lambda x: date_parser(x))
    # gurugram_data_co.index = gurugram_data_co['date']
    # gurugram_data_co = gurugram_data_co.drop(['date'], axis=1)

    # gurugram_data_co = gurugram_data_co['{element}'].resample('D').mean()

    # Aggregate data: take the mean of lat, long, and {element} for each date
    gurugram_data_co = gurugram_data_co.groupby('date').agg({
        'id': 'first',  # Count the number of records for each date
        'lat': 'first',  # You can also use 'first' or 'median' if preferred
        'long': 'first',  # You can also use 'first' or 'median' if preferred
        f'{element}': 'mean'     # Resample CO values by taking the mean
    }).reset_index()

    # Set the date as the index for time series modeling
    gurugram_data_co.set_index('date', inplace=True)


    # Fit the SARIMAX model
    mod = sm.tsa.statespace.SARIMAX(gurugram_data_co[f'{element}'],
                                    order=(1, 0, 1),
                                    seasonal_order=(1, 0, 1, 12),  # Adjust seasonal period if needed
                                    enforce_stationarity=False,
                                    enforce_invertibility=False)
    results = mod.fit()

    # Generate predictions
    gurugram_data_co['predicted_CO'] = results.predict(start=0, end=len(gurugram_data_co) - 1)
    gurugram_data_co.reset_index(inplace=True)

    # Forecast for the next 20 days
    forecast_steps = 20
    forecast_index = pd.date_range(start=gurugram_data_co['date'].iloc[-1] + pd.Timedelta(days=1), periods=forecast_steps, freq='D')
    forecast_values = results.forecast(steps=forecast_steps)

    # Create a DataFrame for the forecast
    forecast_df = pd.DataFrame({
        'id': [None] * forecast_steps,
        'date': forecast_index,
        'lat': gurugram_data_co['lat'].iloc[-1],  # Use the last known latitude
        'long': gurugram_data_co['long'].iloc[-1],  # Use the last known longitude
        f'{element}': [None] * forecast_steps,  # No actual CO values for the forecast
        f'predicted_{element}': forecast_values
    })

    # Combine the original data with the forecast
    final_df = pd.concat([gurugram_data_co, forecast_df], ignore_index=True)
    final_df['location'] = 'Gurugram'
    # Save the final DataFrame to a CSV file
    output_columns = ['location','id', 'lat', 'long', 'date', f'{element}', f'predicted_{element}']
    final_df[output_columns].to_csv(f'gurugram_{element}_predictions_with_forecast.csv', index=False)

    print(f"CSV file 'gurugram_{element}_predictions_with_forecast.csv' has been created.")
    

if __name__ == "__main__":
    pass