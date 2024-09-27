import streamlit as st
import pickle
from text_processing import preprocessing_text
from visual_display import plot_pie_chart

with open('tfidf_vectorizer.pickle', 'rb') as handle:
    tfidf_vectorizer = pickle.load(handle)

with open('log_model.pkl', 'rb') as file:
    model = pickle.load(file)

st.set_page_config(page_title="Text Emotion Detector", page_icon=":sparkles:", layout="wide")
st.sidebar.title("Navigation")
st.sidebar.info("This app analyzes emotions in text. Enter your text and see the predicted emotions!")

st.title("Text Emotion detector: ")

display = '''
Click the 'Analyze' button to see the predicted emotion and other relevant details.\n
Example Sentences \n
Sadness: "I can’t believe I lost my job." \n
Happiness: "Spending time with you always makes me smile." \n
Love: "You are so romantic and make me feel so sweet" \n
Anger: "I hated my job and my boss" \n
Fear: "This is so uncomfortable and it is making me feel uneasy." \n
Surprise: "This is amazing !" \n
'''

with st.container():
    st.subheader("Input Your Text")
    st.write("Enter the text for which you want to analyze the emotion in the text box provided.")
with st.expander("More Information"):
    st.write(display)

user_text = st.text_input("Write down your emotion here.")

if st.button("Submit"):
    if user_text:
        processed_input = tfidf_vectorizer.transform([preprocessing_text(str(user_text))])
        predicted_result = model.predict_proba(processed_input)[0]
        plot_pie_chart(predicted_result)
    else:
        st.write("Write your emotion here :) ")
