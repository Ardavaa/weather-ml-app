import numpy as np
import pickle as pkl
import streamlit as st
import plotly.express as px

with open('classifier.pkl', 'rb') as pickle_in:
    classifier = pkl.load(pickle_in)

with open('scaler.pkl', 'rb') as scaler_file:
    mms = pkl.load(scaler_file)

# prediction function
def prediction(Temperature, Humidity, Wind_Speed, Precipitation, Cloud_Cover, Atmospheric_Pressure, UV_Index, Season, Visibility, Location):
    cloud_cover_mapping = {'overcast': 0, 'partly cloudy': 1, 'clear': 2, 'cloudy': 3}
    season_mapping = {'Winter': 0, 'Summer': 1, 'Spring': 2, 'Autumn': 3}
    location_mapping = {'mountain': 0, 'inland': 1, 'coastal': 2}

    Cloud_Cover = cloud_cover_mapping[Cloud_Cover]
    Season = season_mapping[Season]
    Location = location_mapping[Location]

    num_cols = np.array([[Temperature, Humidity, Wind_Speed, Precipitation, Atmospheric_Pressure, UV_Index, Visibility]])
    num_cols_scaled = mms.transform(num_cols)
    input_data = np.concatenate([num_cols_scaled, [[Cloud_Cover, Season, Location]]], axis=1)

    prediction_proba = classifier.predict_proba(input_data)

    weather_types = ['Cloudy', 'Rainy', 'Snowy', 'Sunny']
    predictions = [(weather_types[i], prediction_proba[0][i]) for i in range(len(weather_types))]
    predictions.sort(key=lambda x: x[1], reverse=True)

    return predictions

# main web function
def main():
    st.set_page_config(page_title="Weather Prediction App", page_icon="⛅")
    st.header("Weather Prediction App")
    st.image('img/weather.jpg', width=600)

    # sidebar input
    with st.sidebar:
        st.header("Input Weather Conditions")
        Temperature = st.number_input("Temperature (°C)", value=23.0, step=1.0)
        Humidity = st.number_input("Humidity (%)", value=50.0, step=1.0)
        Wind_Speed = st.number_input("Wind Speed (km/h)", value=5.0, step=1.0)
        Precipitation = st.slider("Precipitation (%)", min_value=0.0, max_value=110.0, step=0.1, value=12.0)
        Atmospheric_Pressure = st.slider("Atmospheric Pressure (hPa)", min_value=800.0, max_value=1200.0, step=0.3, value=1013.0)
        UV_Index = st.slider("UV Index", min_value=0, max_value=14, value=3)
        Visibility = st.slider("Visibility (km)", min_value=0.0, max_value=20.0, step=0.1, value=10.0)
        Season = st.selectbox('Season', ('Winter', 'Summer', 'Spring', 'Autumn'))
        Cloud_Cover = st.selectbox('Cloud Cover', ('overcast', 'partly cloudy', 'clear', 'cloudy'))
        Location = st.selectbox('Location', ('mountain', 'inland', 'coastal'))

    # when user clicked 'predict' button
    if st.sidebar.button("Predict"):
        predictions = prediction(Temperature, Humidity, Wind_Speed, Precipitation, Cloud_Cover, Atmospheric_Pressure, UV_Index, Season, Visibility, Location)
        st.subheader("Weather Possibilities")
        
        col1, col2 = st.columns(2)

        with col2:
            st.subheader(" ", divider="gray")
            for i, (weather, prob) in enumerate(predictions):
                if i == 0:  # highest probability
                    st.success(f"Most likely weather: {weather} ({prob * 100:.2f}%)")
                else:
                    st.warning(f"Possible weather: {weather} ({prob * 100:.2f}%)")

        with col1:
            st.subheader(" ", divider="gray")
            weather_labels, weather_probs = zip(*predictions)
            fig = px.pie(
                names=weather_labels,
                values=weather_probs,
                title="Weather Confidence Levels Chart"
            )

            st.plotly_chart(fig)

    st.info('Copyright © Ardava Barus - All rights reserved')

if __name__ == '__main__':
    main()
