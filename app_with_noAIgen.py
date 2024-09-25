import streamlit as st
import pickle

with open('tfidf_vectorizer.pickle', 'rb') as handle:
    tfidf_vectorizer = pickle.load(handle)

with open('log_model.pkl', 'rb') as file:
    model = pickle.load(file)

def interpret_result(predicted_result):
    emotions = {
        0: ("You are feeling sad 😟"),
        1: ("You are feeling happy 😁"),
        2: ("You are feeling love 🥰"),
        3: ("You are feeling angry 🤬"),
        4: ("You are feeling scared 😨"),
        5: ("You are feeling surprised 😮"),
    }
    message = emotions.get(predicted_result, ('Unknown', "Emotion not recognized"))
    st.write(message)

st.title("Text Emotion detector: ")

user_text = st.text_input("Write down your emotion here.")

if st.button("Submit"):
    if user_text:
        processed_input = tfidf_vectorizer.transform([user_text])
        predicted_result = model.predict(processed_input)[0]
        interpret_result(predicted_result)
    else:
        st.write("Write your emotion here :) ")
