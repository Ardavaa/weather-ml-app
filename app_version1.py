import numpy as np
import pickle as pkl
import streamlit as st

pickle_in = open('classifier.pkl', 'rb')
classifier = pkl.load(pickle_in)

with open('scaler.pkl', 'rb') as scaler_file:
    mms = pkl.load(scaler_file)

# function to make prediction based on user input 
def prediction(Temperature, Humidity, Wind_Speed, Precipitation, Cloud_Cover, Atmospheric_Pressure, UV_Index, Season, Visibility, Location):
    # label mapping
    cloud_cover_mapping = {'overcast': 0, 'partly cloudy': 1, 'clear': 2, 'cloudy': 3}
    Cloud_Cover = cloud_cover_mapping[Cloud_Cover]

    season_mapping = {'Winter': 0, 'Summer': 1, 'Spring': 2, 'Autumn': 3}
    Season = season_mapping[Season]

    location_mapping = {'mountain': 0, 'inland': 1, 'coastal': 2}
    Location = location_mapping[Location]

    # feature scaling
    num_cols = np.array([[Temperature, Humidity, Wind_Speed, Precipitation, Atmospheric_Pressure, UV_Index, Visibility]])
    num_cols_scaled = mms.transform(num_cols)
    input_data = np.concatenate([num_cols_scaled, [[Cloud_Cover, Season, Location]]], axis=1)

    # make predictions based on probabilities
    prediction_proba = classifier.predict_proba(input_data)  # Get the probability of each class

    # label mapping
    weather_types = ['Cloudy', 'Rainy', 'Snowy', 'Sunny']

    # create a list of (weather, probability) tuples
    predictions = [(weather_types[i], prediction_proba[0][i]) for i in range(len(weather_types))]

    # sort by probability, highest first
    predictions.sort(key=lambda x: x[1], reverse=True)

    return predictions

# main functiofor webpage
def main():
    # page config
    st.set_page_config(page_title="Weather Prediction App", page_icon="⛅", layout="centered")

    # discord-like look
    st.markdown(
        """
        <style>
        body {
            background-color: #2C2F33;
            color: #ffffff;
            font-family: 'Segoe UI', sans-serif;
        }
        .stApp {
            background-color: #2C2F33;
        }
        .main-title {
            color: #7289DA;
            text-align: center;
            font-size: 36px;
            font-weight: 600;
        }
        .input-container {
            padding: 20px;
            border-radius: 10px;
            background-color: #23272A;
            box-shadow: 0px 0px 10px rgba(0, 0, 0, 0.5);
        }
        .stButton>button {
            background-color: #5865F2;
            color: #ffffff;
            border-radius: 8px;
            font-size: 16px;
            font-weight: bold;
            height: 45px;
            width: 100%;
            margin-top: 15px;
        }
        .stButton>button:hover {
            background-color: #4d5bd4;
            color: #ffffff;
        }
        .result {
            margin-top: 20px;
            padding: 15px;
            border-radius: 8px;
            font-size: 18px;
            font-weight: 500;
        }
        .result.green {
            background-color: #43b581;
            color: white;
        }
        .result.yellow {
            background-color: #F0AD4E;
            color: black;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # web-title
    st.markdown("<div class='main-title'>Weather Prediction ML App</div>", unsafe_allow_html=True)

    # input container
    with st.form("prediction_form"):

        # input fields
        Temperature = st.number_input("Temperature (°C)", value=None, step=1.0, placeholder='Enter the temperature in Celsius')
        Humidity = st.number_input("Humidity (%)", value=None, step=1.0, placeholder='Enter the humidity in percentage')
        Wind_Speed = st.number_input("Wind Speed (km/h)", value=None, step=1.0, placeholder='Enter the wind speed in km/h')
        
        Precipitation = st.slider("Precipitation (%)", min_value=0.0, max_value=110.0, step=0.1)
        Atmospheric_Pressure = st.slider("Atmospheric Pressure (hPa)", min_value=800.0, max_value=1200.0, step=0.3)
        UV_Index = st.slider("UV Index", min_value=0, max_value=14)
        Visibility = st.slider("Visibility (km)", min_value=0.0, max_value=20.0, step=0.1)
        
        
        Season = st.selectbox('Season', ('Winter', 'Summer', 'Spring', 'Autumn'))
        Cloud_Cover = st.selectbox('Cloud Cover', ('overcast', 'partly cloudy', 'clear', 'cloudy'))
        Location = st.selectbox('Location', ('mountain', 'inland', 'coastal'))

        # predict button
        submitted = st.form_submit_button("Predict")
        st.markdown("</div>", unsafe_allow_html=True)

    # make the prediction and display results when 'Predict' is clicked
    if submitted:
        predictions = prediction(Temperature, Humidity, Wind_Speed, Precipitation, Cloud_Cover, Atmospheric_Pressure, UV_Index, Season, Visibility, Location)

        # display the probabilities with appropriate background color
        for i, (weather, prob) in enumerate(predictions):
            if i == 0:  # highest probability
                st.markdown(f"<div class='result green'>The weather is {weather} with a confidence of {prob*100:.2f}%</div>", unsafe_allow_html=True)
            else:
                st.markdown(f"<div class='result yellow'>The weather is {weather} with a confidence of {prob*100:.2f}%</div>", unsafe_allow_html=True)

if __name__ == '__main__':
    main()

