import streamlit as st
import pickle
from preprocessing import *
from nearest_cluster import recommend_songs
from visual_display import plot_pie_chart

# ------ Processing
with open('tfidf_vectorizer.pickle', 'rb') as handle:
    tfidf_vectorizer = pickle.load(handle)

# ------ load trained model
with open('log_model.pkl', 'rb') as file:
    logmodel = pickle.load(file)

def display_music(track_ids):
    iframe_str = ""
    for track_id in track_ids:
        iframe_str += f"""
        <iframe src="https://open.spotify.com/embed/track/{track_id}" width="300" height="80" frameBorder="0" allowtransparency="true" allow="encrypted-media"></iframe>
        """
    st.sidebar.markdown(iframe_str, unsafe_allow_html=True)

st.set_page_config(page_title="Text Emotion Detector", page_icon=":sparkles:", layout="wide")

# --------------------- Side Bar --------------------- #
st.sidebar.title("🎶 Welcome to My Music Recommendation App!")
st.sidebar.markdown("""
### 🎤 Recommendations Based on Your Emotion
""")
# ---------------------------------------------------- #

st.title("Emotion Text Detector: ")

with st.container():
    st.write("Enter the text for which you want to analyze the emotion in the text box provided.")
user_text = st.text_input("Write down your text here.")

if st.button("Submit"):
    if user_text:
        processed_input = tfidf_vectorizer.transform([preprocessing_text(str(user_text))])
        predicted_result = logmodel.predict_proba(processed_input)
        plot_pie_chart(predicted_result[0])
        map_emotion = map_emotionresults(dict(zip(logmodel.classes_,predicted_result[0])))
        track_ids = recommend_songs(emotion_to_features(map_emotion), num_recommendations=5)
        display_music(track_ids)
    else:
        st.write("Write your emotion here :) ")

display = """
### How It Works
This application uses text emotion analysis to recommend songs that resonate with your current feelings. Simply type in an emotion, and we'll provide a selection of songs to match.

### Examples of Emotions:
- Happiness:    Spending time with you always makes me smile.
- Sadness:      I can’t believe I lost my job.
- Love:         You are so romantic and make me feel so sweet
- Anger:        I hate my neighbours pets
- Fear:         I am having nightmares among everyday
- Surprise:     This is amazing! How did you do that!
### What to Expect:
After entering your emotion, you will see a list of songs that are curated based on your input.
Enjoy the music and let it uplift your spirits!
"""
with st.expander("More Information"):
    st.write(display)
