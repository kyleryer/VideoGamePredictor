import streamlit as st
import pandas as pd
import joblib

# Load model
model = joblib.load('video_game_prediction_model.pkl')

# Set app title and description
st.markdown("<h1 style='text-align: center;'>Video Game Hit Predictor</h1>", unsafe_allow_html=True)
st.write("Enter some potential ideas for a game to predict if it's likely to be a 'Hit' (selling over 1 million copies globally) or not.")

# Map platform abbreviations to actual platform names
platform_map = {
    '2600': 'Atari 2600',
    'WS': 'Bandai WonderSwan',
    'XB': 'Microsoft Xbox',
    'X360': 'Microsoft Xbox 360',
    'XOne': 'Microsoft Xbox One',
    'PCFX': 'NEC PC-FX',
    'NG': 'Neo Geo',
    '3DS': 'Nintendo 3DS',
    'N64': 'Nintendo 64',
    'DS': 'Nintendo DS',
    'NES': 'Nintendo Entertainment System (NES)',
    'GB': 'Nintendo Game Boy',
    'GBA': 'Nintendo Game Boy Advance',
    'GC': 'Nintendo GameCube',
    'Wii': 'Nintendo Wii',
    'WiiU': 'Nintendo WiiU',
    '3DO': 'Panasonic 3DO',
    'PC': 'PC',
    'SCD': 'Sega CD',
    'DC': 'Sega Dreamcast',
    'GG': 'Sega Game Gear',
    'GEN': 'Sega Genesis',
    'SAT': 'Sega Saturn',
    'PS': 'Sony PlayStation',
    'PS2': 'Sony PlayStation 2',
    'PS3': 'Sony PlayStation 3',
    'PS4': 'Sony PlayStation 4',
    'PSP': 'Sony PlayStation Portable (PSP)',
    'PSV': 'Sony PlayStation Vita',
    'SNES': 'Super Nintendo Entertainment System (SNES)',
    'TG16': 'TurboGrafx-16'
}

# Create genre list
genres = [
    'Action',
    'Adventure',
    'Fighting',
    'Misc',
    'Platform',
    'Puzzle',
    'Racing',
    'Role-Playing',
    'Shooter',
    'Simulation',
    'Sports',
    'Strategy'
]

# Create input fields for user to enter data
st.subheader("Game Features")
col1, col2, col3 = st.columns(3)

with col1:
    platform_selection = st.selectbox("Platform", list(platform_map.keys()), format_func=lambda x: platform_map[x]) # User will see written out platform name, machine

with col2:
    genre_selection = st.selectbox("Genre", genres)

with col3:
    publisher_selection = st.selectbox("Publisher Type (choose Top if you are a well known publisher and choose Unknown if you are not)", ['Top', 'Unknown'])

st.write("---")

# Style predict button
st.markdown("""
<style>
div.stButton > button:first-child {
    background-color: #0099ff;
    color: white;
    font-size: 20px;
    font-weight: bold;
    height: 2.5em;
    width: 100%;
}
</style>
""", unsafe_allow_html=True)

# Button to trigger prediction
if st.button("Predict"):
    # Prepare the input data
    input_data = pd.DataFrame({
        'Platform': [platform_selection],
        'Genre': [genre_selection],
        'Publisher_Type': [publisher_selection]
    })

    # Get hit probability
    hit_probability = model.predict_proba(input_data)[0][1]

    # Use thresholds for hit probability so that different publisher types are balanced
    if publisher_selection == 'Top':
        threshold_high = 0.5
        threshold_low = 0.35
    else:
        threshold_high = 0.35
        threshold_low = 0.2

    # Display results
    st.subheader("Prediction Result")
    if hit_probability >= threshold_high:
        st.success(f"This game is likely to be a **HIT**!")
    elif hit_probability >= threshold_low:
        st.warning(f"This game could **potentially** be a hit!")
    else:
        st.error(f"This game is more than likely not going to be a hit.")
    
    st.write(f"Confidence (Probability of being a Hit):")
    st.markdown(f"<h1 style='text-align: center; color: #0099ff;'>{hit_probability:.1%}</h1>", unsafe_allow_html=True)
    st.info(f"Analysis is based on a/an **{publisher_selection}** publisher. If you are an unknown publisher, it is less likely to reach 1 million sales even with a hit idea.")
    st.info(f"A hit probability of 35% or more as an unknown publisher indicates a likely hit.")