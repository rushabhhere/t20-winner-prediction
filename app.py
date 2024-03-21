import streamlit as st
import pickle
import joblib
import pandas as pd
import sklearn
from sklearn.preprocessing import MinMaxScaler, StandardScaler

teams = [
    "Sunrisers Hyderabad",
    "Mumbai Indians",
    "Royal Challengers Bangalore",
    "Kolkata Knight Riders",
    "Kings XI Punjab",
    "Chennai Super Kings",
    "Rajasthan Royals",
    "Delhi Capitals",
]

cities = [
    "Hyderabad",
    "Bangalore",
    "Mumbai",
    "Indore",
    "Kolkata",
    "Delhi",
    "Chandigarh",
    "Jaipur",
    "Chennai",
    "Cape Town",
    "Port Elizabeth",
    "Durban",
    "Centurion",
    "East London",
    "Johannesburg",
    "Kimberley",
    "Bloemfontein",
    "Ahmedabad",
    "Cuttack",
    "Nagpur",
    "Dharamsala",
    "Visakhapatnam",
    "Pune",
    "Raipur",
    "Ranchi",
    "Abu Dhabi",
    "Sharjah",
    "Mohali",
    "Bengaluru",
]

model = pickle.load(open("./models/logistic_regression.pkl", "rb"))
scaler = joblib.load(open("./models/scaler.pkl", "rb"))
st.title("IPL Win Predictor")

col1, col2 = st.columns(2)

with col1:
    batting_team = st.selectbox("Select the Batting team", sorted(teams))

with col2:
    bowling_team = st.selectbox("Select the Bowling team", sorted(teams))

selected_city = st.selectbox("Select host city", sorted(cities))

target = st.number_input("Target")

col3, col4, col5 = st.columns(3)

with col3:
    score = st.number_input("Score")
with col4:
    overs = st.number_input("Overs completed")
with col5:
    wickets = st.number_input("Wickets out")

if st.button("Predict Probability"):
    runs_left = target - score
    balls_left = 120 - (overs * 6)
    wickets = 10 - wickets
    crr = score / overs
    rrr = (runs_left * 6) / balls_left

    # input_df = pd.DataFrame(
    #     {
    #         "batting_team": [batting_team],
    #         "bowling_team": [bowling_team],
    #         "city": [selected_city],
    #         "runs_left": [runs_left],
    #         "balls_left": [balls_left],
    #         "wickets": [wickets],
    #         "total_runs_x": [target],
    #         "crr": [crr],
    #         "rrr": [rrr],
    #     }
    # )

    input_df = [[runs_left, wickets, crr, rrr, balls_left]]
    st.dataframe(input_df)
    input_df = scaler.transform(input_df)
    st.dataframe(input_df)
    result = model.predict_proba(input_df)
    st.write(result)
    loss = result[0][0]
    win = result[0][1]
    st.header(batting_team + "- " + str(round(win * 100)) + "%")
    st.header(bowling_team + "- " + str(round(loss * 100)) + "%")
