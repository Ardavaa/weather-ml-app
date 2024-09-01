import numpy as np
import pickle as pkl
import streamlit as st

# loading the trained model
pickle_in = open('classifier.pkl', 'rb')
classifier = pkl.load(pickle_in)

# loading the saved MinMaxScaler
with open('scaler.pkl', 'rb') as scaler_file:
    mms = pkl.load(scaler_file)

# defining the function which will make the prediction using the data which the user inputs 
def prediction(Temperature, Humidity, Wind_Speed, Precipitation, Cloud_Cover, Atmospheric_Pressure, UV_Index, Season, Visibility, Location):
    # pre-processing user input

    # Cloud Cover
    if Cloud_Cover == 'overcast':
        Cloud_Cover = 0
    elif Cloud_Cover == 'partly cloudy':
        Cloud_Cover = 1
    elif Cloud_Cover == 'clear':
        Cloud_Cover = 2
    elif Cloud_Cover == 'cloudy':
        Cloud_Cover = 3
    
    # Season
    if Season == 'Winter':
        Season = 0
    elif Season == 'Summer':
        Season = 1
    elif Season == 'Spring':
        Season = 2
    elif Season == 'Autumn':
        Season = 3

    # Location
    if Location == 'mountain':
        Location = 0
    elif Location == 'inland':
        Location = 1
    elif Location == 'coastal':
        Location = 2

    # feature scaling num_cols
    num_cols = np.array([[Temperature, Humidity, Wind_Speed, Precipitation, Atmospheric_Pressure, UV_Index, Visibility]])
    num_cols_scaled = mms.transform(num_cols)

    input_data = np.concatenate([num_cols_scaled, [[Cloud_Cover, Season, Location]]], axis=1)

    # making predictions
    prediction_proba = classifier.predict_proba(input_data)  # get the probability of each class

    # label mapping
    weather_types = ['Cloudy', 'Rainy', 'Snowy', 'Sunny']

    # create a list of (weather, probability) tuples
    predictions = [(weather_types[i], prediction_proba[0][i]) for i in range(len(weather_types))]
    
    # sort by probability, highest first
    predictions.sort(key=lambda x: x[1], reverse=True)

    return predictions

# this is the main function in which we define our webpage
def main():
    # front end elements of the web page
    html_temp = """
    <div style ="background-color:yellow;padding:13px">
    <h1 style ="color:black;text-align:center;">Weather Prediction ML App</h1>
    </div>
    """

    st.markdown(html_temp, unsafe_allow_html=True)

    # following lines create boxes in which user can enter data required to make prediction
    # number input
    Temperature = st.number_input("Temperature", value=None, step=1.0, placeholder='Enter the temperature in Celsius')
    Humidity = st.number_input("Humidity",value=None,  step=1.0, placeholder='Enter the humidity in percentage')
    Wind_Speed = st.number_input("Wind Speed", value=None, step=1.0, placeholder='Enter the wind speed in km/h')
    
    # slider input
    Atmospheric_Pressure = st.slider("Atmospheric Pressure", min_value=800.0, max_value=1200.0, step=0.3)
    Precipitation = st.slider("Precipitation (%)", min_value=0.0, max_value=110.0, step=0.1)
    UV_Index = st.slider("UV Index", min_value=0, max_value=14)
    Season = st.selectbox('Season', ('Winter', 'Summer', 'Spring', 'Autumn'))

    # selectbox input
    Visibility = st.slider("Visibility (km)", min_value=0.0, max_value=20.0, step=0.1)
    Cloud_Cover = st.selectbox('Cloud Cover', ('overcast', 'partly cloudy', 'clear', 'cloudy'))
    Location = st.selectbox('Location', ('mountain', 'inland', 'coastal'))
    
    # when 'Predict' is clicked, make the prediction and store it
    if st.button("Predict"):
        predictions = prediction(Temperature, Humidity, Wind_Speed, Precipitation, Cloud_Cover, Atmospheric_Pressure, UV_Index, Season, Visibility, Location)
        
        # display the probabilities with appropriate background color
        for i, (weather, prob) in enumerate(predictions):
            if i == 0:  # highest probability
                color = 'background-color:green; color:white;'
            else:
                color = 'background-color:yellow; color:black;'
            st.markdown(f"<div style='{color} padding: 10px;'>The weather is {weather} with a confidence of {prob*100:.2f}%</div>", unsafe_allow_html=True)

if __name__ == '__main__':
    main()
