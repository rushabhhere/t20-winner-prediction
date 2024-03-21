import streamlit as st
import pickle
import joblib
import warnings

warnings.filterwarnings("ignore")

# load keras model
from tensorflow.keras.saving import load_model

# from tensorflow.keras.models import load_model
from matches import matches, get_match_details
from plotly import express as px

st.markdown(
    """
<style>
.stProgress > div > div > div > div {
    background-color: #77DD76; /* Change the color here */
    border-radius: 0px; /* Change the border radius here */
}
.stProgress > div > div > div {
    height: 20px; /* Change the height to adjust thickness */
    background-color: #FF6962; /* Change the color here */
}
</style>
""",
    unsafe_allow_html=True,
)

modelpath = {
    "Logistic Regression": "./models/logistic_regression.pkl",
    "Random Forest": "./models/random_forest.pkl",
    "Neural Network": "./models/fnn_1l_model.keras",
}

st.title("T20 International Win Predictor")

# model selector
selection = st.selectbox(
    "Select Model", ["Logistic Regression", "Random Forest", "Neural Network"]
)

model = pickle.load(open(modelpath["Logistic Regression"], "rb"))

if selection in ["Logistic Regression", "Random Forest"]:
    model = pickle.load(open(modelpath[selection], "rb"))
else:
    model = load_model(modelpath[selection])
    # model = joblib.load(open(modelpath[selection], "rb"))

scaler = joblib.load(open("./models/scaler.pkl", "rb"))

tab1, tab2 = st.tabs(["Predict", "Historical"])

with tab1:
    st.markdown("### Prediction")

    teams = [
        "Mozambique",
        "New Zealand",
        "Ireland",
        "Tanzania",
        "Namibia",
        "Qatar",
        "Turkey",
        "Rwanda",
        "Finland",
        "South Africa",
        "Sri Lanka",
        "Singapore",
        "England",
        "Kuwait",
        "Gambia",
        "Zimbabwe",
        "Australia",
        "Netherlands",
        "Thailand",
        "Malaysia",
        "Bhutan",
        "Afghanistan",
        "Malta",
        "Nepal",
        "United Arab Emirates",
        "Spain",
        "Gibraltar",
        "West Indies",
        "India",
        "Bangladesh",
        "Ghana",
        "Nigeria",
        "Denmark",
        "Czech Republic",
        "Romania",
        "Austria",
        "Vanuatu",
        "Bahamas",
        "Bermuda",
        "Jersey",
        "Italy",
        "Bahrain",
        "Hungary",
        "Sweden",
        "Saudi Arabia",
        "Seychelles",
        "Guernsey",
        "Oman",
        "Uganda",
        "Pakistan",
        "Kenya",
        "Switzerland",
        "Papua New Guinea",
        "Luxembourg",
        "Hong Kong",
        "Cayman Islands",
        "Philippines",
        "Sierra Leone",
        "Bulgaria",
        "Scotland",
        "Serbia",
        "Germany",
        "Canada",
        "China",
        "Panama",
        "Myanmar",
        "South Korea",
        "Belgium",
        "Maldives",
        "United States of America",
        "Botswana",
        "Lesotho",
        "France",
        "Norway",
        "Argentina",
        "Fiji",
        "Japan",
        "Cambodia",
        "Belize",
        "Indonesia",
        "St Helena",
        "Isle of Man",
        "Malawi",
        "Mongolia",
        "Cameroon",
        "Samoa",
        "Eswatini",
        "Estonia",
        "Cook Islands",
        "Greece",
        "Portugal",
        "Slovenia",
        "Mexico",
        "Croatia",
        "Cyprus",
        "Mali",
        "Israel",
    ]

    venues = [
        "Gahanga International Cricket Stadium. Rwanda",
        "Pallekele International Cricket Stadium",
        "Al Amerat Cricket Ground Oman Cricket (Ministry Turf 1)",
        "Willowmoore Park, Benoni",
        "Wanderers Cricket Ground, Windhoek",
        "West End Park International Cricket Stadium, Doha",
        "Shere Bangla National Stadium, Mirpur",
        "Tikkurila Cricket Ground",
        "Tafawa Balewa Square Cricket Oval, Lagos",
        "Svanholm Park, Brondby",
        "New Wanderers Stadium",
        "Eden Gardens",
        "Queens Sports Club, Bulawayo",
        "Saxton Oval",
        "Bharat Ratna Shri Atal Bihari Vajpayee Ekana Cricket Stadium, Lucknow",
        "Central Broward Regional Park Stadium Turf Ground",
        "Integrated Polytechnic Regional Centre",
        "Sharjah Cricket Stadium",
        "Punjab Cricket Association IS Bindra Stadium, Mohali, Chandigarh",
        "The Village, Malahide",
        "UKM-YSD Cricket Oval, Bangi",
        "Bayuemas Oval, Kuala Lumpur",
        "Harare Sports Club",
        "Marsa Sports Club",
        "Sheikh Zayed Stadium",
        "Gymkhana Club Ground, Nairobi",
        "Desert Springs Cricket Ground, Almeria",
        "Europa Sports Complex",
        "Central Broward Regional Park Stadium Turf Ground, Lauderhill",
        "Beausejour Stadium, Gros Islet",
        "Punjab Cricket Association IS Bindra Stadium, Mohali",
        "National Stadium",
        "United Cricket Club Ground, Windhoek",
        "Scott Page Field, Vinor",
        "Civil Service Cricket Club, Stormont, Belfast",
        "Tolerance Oval",
        "Warner Park, St Kitts",
        "Dr. Y.S. Rajasekhara Reddy ACA-VDCA Cricket Stadium",
        "Kerava National Cricket Ground",
        "Kinrara Academy Oval",
        "Hurlingham Club Ground, Buenos Aires",
        "Dubai International Cricket Stadium",
        "Seddon Park",
        "Adelaide Oval",
        "Trent Bridge",
        "Tribhuvan University International Cricket Ground, Kirtipur",
        "Moara Vlasiei Cricket Ground, Ilfov County",
        "Brisbane Cricket Ground, Woolloongabba, Brisbane",
        "Bayer Uerdingen Cricket Ground",
        "Al Amerat Cricket Ground Oman Cricket (Ministry Turf 2)",
        "Eden Park",
        "GMHBA Stadium, South Geelong, Victoria",
        "Shere Bangla National Stadium",
        "Rangiri Dambulla International Stadium",
        "National Cricket Stadium, St George's, Grenada",
        "Brisbane Cricket Ground, Woolloongabba",
        "College Field",
        "R.Premadasa Stadium, Khettarama",
        "SuperSport Park",
        "Wankhede Stadium",
        "R Premadasa Stadium, Colombo",
        "ICC Academy",
        "Rajiv Gandhi International Cricket Stadium, Dehradun",
        "The Wanderers Stadium",
        "Zahur Ahmed Chowdhury Stadium, Chattogram",
        "R Premadasa Stadium",
        "National Stadium, Hamilton",
        "Kennington Oval",
        "Amini Park, Port Moresby",
        "Moara Vlasiei Cricket Ground",
        "Independence Park, Port Vila",
        "Mulpani Cricket Ground",
        "Hazelaarweg",
        "Belgrano Athletic Club Ground, Buenos Aires",
        "Indian Association Ground",
        "Himachal Pradesh Cricket Association Stadium",
        "County Ground",
        "National Sports Academy, Sofia",
        "Royal Brussels Cricket Club Ground, Waterloo",
        "University of Lagos Cricket Oval",
        "Manuka Oval",
        "Bay Oval",
        "Queen's Park Oval, Port of Spain",
        "The Rose Bowl",
        "Sportpark Westvliet",
        "Sportpark Het Schootsveld",
        "Lisicji Jarak Cricket Ground",
        "Kensington Oval, Bridgetown",
        "M Chinnaswamy Stadium",
        "Old Trafford",
        "Grange Cricket Club Ground, Raeburn Place, Edinburgh",
        "VRA Ground",
        "Zayed Cricket Stadium, Abu Dhabi",
        "La Manga Club Bottom Ground",
        "Civil Service Cricket Club, Stormont",
        "Kingsmead",
        "Vidarbha Cricket Association Stadium, Jamtha",
        "Gucherre Cricket Ground",
        "Greater Noida Sports Complex Ground",
        "Mahinda Rajapaksa International Cricket Stadium, Sooriyawewa",
        "Western Australia Cricket Association Ground",
        "Gaddafi Stadium, Lahore",
        "Arun Jaitley Stadium",
        "White Hill Field, Sandys Parish",
        "Windsor Park, Roseau",
        "Zahur Ahmed Chowdhury Stadium",
        "Perth Stadium",
        "Terdthai Cricket Ground, Bangkok",
        "Sophia Gardens, Cardiff",
        "The Wanderers Stadium, Johannesburg",
        "John Davies Oval, Queenstown",
        "Bharat Ratna Shri Atal Bihari Vajpayee Ekana Cricket Stadium",
        "Sabina Park, Kingston, Jamaica",
        "Coolidge Cricket Ground, Antigua",
        "Gahanga International Cricket Stadium, Rwanda",
        "Lord's",
        "Providence Stadium, Guyana",
        "Westpac Stadium",
        "St Albans Club, Buenos Aires",
        "Sano International Cricket Ground",
        "Amini Park",
        "Kingsmead, Durban",
        "Meersen, Gent",
        "Wankhede Stadium, Mumbai",
        "Grange Cricket Club Ground, Raeburn Place",
        "Sheikh Abu Naser Stadium",
        "Narendra Modi Stadium",
        "United Cricket Club Ground",
        "The Rose Bowl, Southampton",
        "Daren Sammy National Cricket Stadium, Gros Islet, St Lucia",
        "Achimota Senior Secondary School A Field, Accra",
        "Dr. Y.S. Rajasekhara Reddy ACA-VDCA Cricket Stadium, Visakhapatnam",
        "Hagley Oval, Christchurch",
        "National Stadium, Karachi",
        "Rawalpindi Cricket Stadium",
        "Bellerive Oval, Hobart",
        "Melbourne Cricket Ground",
        "King George V Sports Ground",
        "Mission Road Ground, Mong Kok, Hong Kong",
        "Entebbe Cricket Oval",
        "St Georges Quilmes",
        "Tony Ireland Stadium",
        "Barsapara Cricket Stadium, Guwahati",
        "County Ground, Bristol",
        "Indian Association Ground, Singapore",
        "Newlands",
        "Kensington Oval, Bridgetown, Barbados",
        "ICC Global Cricket Academy",
        "Malahide, Dublin",
        "Vidarbha Cricket Association Stadium, Jamtha, Nagpur",
        "GB Oval, Szodliget",
        "Maple Leaf North-West Ground",
        "Grange Cricket Club, Raeburn Place",
        "Himachal Pradesh Cricket Association Stadium, Dharamsala",
        "Queens Sports Club",
        "AMI Stadium",
        "Bay Oval, Mount Maunganui",
        "Jade Stadium",
        "Headingley, Leeds",
        "St George's Park",
        "Sylhet Stadium",
        "The Village, Malahide, Dublin",
        "Eden Park, Auckland",
        "Providence Stadium",
        "Rajiv Gandhi International Cricket Stadium",
        "Sportpark Het Schootsveld, Deventer",
        "Bready",
        "Bulawayo Athletic Club",
        "Gymkhana Club Ground",
        "Terdthai Cricket Ground",
        "M Chinnaswamy Stadium, Bengaluru",
        "Lugogo Cricket Oval",
        "JSCA International Stadium Complex, Ranchi",
        "Stadium Australia",
        "Tribhuvan University International Cricket Ground",
        "Edgbaston, Birmingham",
        "Sydney Cricket Ground",
        "McLean Park",
        "Gaddafi Stadium",
        "Feroz Shah Kotla",
        "Barsapara Cricket Stadium",
        "Greenfield International Stadium",
        "Sir Vivian Richards Stadium, North Sound, Antigua",
        "Shaheed Veer Narayan Singh International Stadium, Raipur",
        "Zhejiang University of Technology Cricket Field",
        "Buffalo Park",
        "Edgbaston",
        "College Field, St Peter Port",
        "ICC Academy Ground No 2",
        "Kyambogo Cricket Oval",
        "Sylhet International Cricket Stadium",
        "Sportpark Westvliet, The Hague",
        "Bermuda National Stadium",
        "Happy Valley Ground",
        "Maharashtra Cricket Association Stadium, Pune",
        "Warner Park, Basseterre",
        "JSCA International Stadium Complex",
        "Sophia Gardens",
        "SuperSport Park, Centurion",
        "Brian Lara Stadium, Tarouba, Trinidad",
        "Eden Gardens, Kolkata",
        "Senwes Park",
        "Barabati Stadium",
        "Green Park",
        "Warner Park, Basseterre, St Kitts",
        "Hagley Oval",
        "Clontarf Cricket Club Ground",
        "Trent Bridge, Nottingham",
        "Rajiv Gandhi International Stadium, Uppal",
        "Goldenacre, Edinburgh",
        "ICC Academy, Dubai",
        "Wanderers",
        "Punjab Cricket Association Stadium, Mohali",
        "Manuka Oval, Canberra",
        "Bready Cricket Club, Magheramason",
        "M.Chinnaswamy Stadium",
        "Bready Cricket Club, Magheramason, Bready",
        "MA Chidambaram Stadium, Chepauk",
        "McLean Park, Napier",
        "Sardar Patel Stadium, Motera",
        "Khan Shaheb Osman Ali Stadium",
        "Windsor Park, Roseau, Dominica",
        "Udayana Cricket Ground",
        "Sir Vivian Richards Stadium, North Sound",
        "Riverside Ground",
        "University Oval",
        "Solvangs Park, Glostrup",
        "Saurashtra Cricket Association Stadium, Rajkot",
        "Achimota Senior Secondary School B Field, Accra",
        "Mission Road Ground, Mong Kok",
        "OUTsurance Oval",
        "Holkar Cricket Stadium, Indore",
        "Santarem Cricket Ground",
        "Sportpark Maarschalkerweerd",
        "Sky Stadium, Wellington",
        "Pierre Werner Cricket Ground",
        "Castle Avenue, Dublin",
        "Arnos Vale Ground, Kingstown",
        "Old Trafford, Manchester",
        "University Oval, Dunedin",
        "Saurashtra Cricket Association Stadium",
        "Narendra Modi Stadium, Ahmedabad",
        "Arun Jaitley Stadium, Delhi",
        "Desert Springs Cricket Ground",
        "Sabina Park, Kingston",
        "National Cricket Stadium, Grenada",
        "Subrata Roy Sahara Stadium",
        "University of Doha for Science and Technology",
        "Sky Stadium",
        "Greenfield International Stadium, Thiruvananthapuram",
        "Holkar Cricket Stadium",
        "Carrara Oval",
        "P Sara Oval",
        "Brabourne Stadium",
        "Riverside Ground, Chester-le-Street",
        "Maharashtra Cricket Association Stadium",
        "Barabati Stadium, Cuttack",
        "Mangaung Oval",
        "Sawai Mansingh Stadium, Jaipur",
        "Moses Mabhida Stadium",
        "Mombasa Sports Club Ground",
        "Simonds Stadium, South Geelong",
        "Bellerive Oval",
        "Boland Park",
        "Rajiv Gandhi International Stadium, Uppal, Hyderabad",
        "Darren Sammy National Cricket Stadium, St Lucia",
        "Wanderers Cricket Ground",
        "De Beers Diamond Oval",
        "Seddon Park, Hamilton",
        "St George's Park, Gqeberha",
    ]

    col1, col2 = st.columns(2)

    with col1:
        batting_team = st.selectbox("Select the Batting team", sorted(teams))

    teams.remove(batting_team)

    with col2:
        bowling_team = st.selectbox("Select the Bowling team", sorted(teams))

    selected_city = st.selectbox("Select venue", sorted(venues))

    target = st.number_input("Target", min_value=0, value=100)

    col3, col4, col5 = st.columns(3)

    with col3:
        score = st.number_input("Score", min_value=0, max_value=(target - 1))
    with col4:
        balls_faced = st.number_input("Balls faced", min_value=0)
    with col5:
        wickets = st.number_input("Wickets out", min_value=0, max_value=9)

    if st.button("Predict Probability"):
        runs_left = target - score
        balls_left = 120 - balls_faced
        wickets = 10 - wickets
        crr = 0
        rrr = target / 20

        if balls_left == 120:
            crr = score
        else:
            crr = (score * 6) / (120 - balls_left)

        if balls_left == 0:
            rrr = runs_left / (1 / 6)
        else:
            rrr = runs_left / (balls_left / 6)

        input_df = [[runs_left, wickets, crr, rrr, balls_left]]
        input_df = scaler.transform(input_df)

        result = 1
        batting_team_probability = 0.9
        bowling_team_probability = 0.1

        if selection in ["Logistic Regression", "Random Forest"]:
            result = model.predict_proba(input_df)
            batting_team_probability = result[0][1]
            bowling_team_probability = 1 - batting_team_probability
        else:
            result = model.predict(input_df)
            batting_team_probability = float(result[0][0])
            bowling_team_probability = 1 - batting_team_probability

        print(selection + ": " + str(batting_team_probability))
        progress = st.progress(batting_team_probability)

        cols = st.columns(2)
        with cols[0]:
            st.markdown(
                batting_team
                + "- "
                + str(round(batting_team_probability * 100, 2))
                + "%"
            )
        with cols[1]:
            st.markdown(
                f'<p style="text-align:right">{bowling_team} - {str(round(bowling_team_probability * 100, 2))}% </p>',
                unsafe_allow_html=True,
            )

with tab2:
    st.markdown("### Match Details")

    match = st.selectbox(
        "Select the Match", sorted([match.split("=")[1] for match in matches])
    )

    idx = [match.split("=")[1] for match in matches].index(match)
    match_id = matches[idx].split("=")[0]

    match_df = get_match_details(match_id)

    predictions = []
    batting_team_t2 = match_df.iloc[0]["batting_team"]
    bowling_team_t2 = match_df.iloc[0]["bowling_team"]

    key1 = f"{batting_team_t2}"
    key2 = f"{bowling_team_t2}"

    for i in range(len(match_df)):
        runs_left = match_df.iloc[i]["runs_required"]
        runs_scored = match_df.iloc[i]["runs_scored"]
        wickets = 10 - match_df.iloc[i]["wickets_remaining"]
        crr = match_df.iloc[i]["crr"]
        rrr = match_df.iloc[i]["rrr"]
        balls_left = match_df.iloc[i]["balls_remaining"]
        balls_faced = 120 - balls_left
        batting_team_t2 = match_df.iloc[i]["batting_team"]
        bowling_team_t2 = match_df.iloc[i]["bowling_team"]
        is_wicket = type(match_df.iloc[i]["wicket_type"]) == str

        input_df = [[runs_left, wickets, crr, rrr, balls_left]]
        input_df = scaler.transform(input_df)

        result = 1
        batting_team_probability = 0.9
        bowling_team_probability = 0.1

        if selection in ["Logistic Regression", "Random Forest"]:
            result = model.predict_proba(input_df)
            batting_team_probability = result[0][1]
            bowling_team_probability = 1 - batting_team_probability
        else:
            result = model.predict(input_df)
            batting_team_probability = result[0][0]
            bowling_team_probability = 1 - batting_team_probability

        predictions.append(
            {
                "Runs Scored": runs_scored,
                "Wickets": wickets,
                "Current RR": crr,
                "Required RR": rrr,
                "Balls Faced": balls_faced,
                key1: round(batting_team_probability, 2),
                key2: round(bowling_team_probability, 2),
                "wicket": is_wicket,
            }
        )

    # plot predictions as an interactive line chart
    df = px.data.tips()
    fig = px.line(
        predictions,
        x="Balls Faced",
        y=[key1, key2],
        hover_data={
            "Runs Scored": True,
            "Wickets": True,
            "Current RR": True,
            "Required RR": True,
            "variable": False,
        },
        title="Win Probability",
        labels={"value": "Win Probability", "variable": "Team"},
    )

    # y must go from 0 to 1
    fig.update_yaxes(range=[0, 1])

    # indicate wickets with red 'x' markers
    for i in range(len(predictions)):
        if predictions[i]["wicket"]:
            fig.add_shape(
                type="line",
                x0=predictions[i]["Balls Faced"],
                y0=0,
                x1=predictions[i]["Balls Faced"],
                y1=1,
                line=dict(color="red", width=2),
            )

    st.plotly_chart(fig)

    st.write(match_df)
