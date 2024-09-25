import streamlit as st
import pickle
import openai
import requests
from PIL import Image

openai.api_key = 'API_KEY'

with open('tfidf_vectorizer.pickle', 'rb') as handle:
    tfidf_vectorizer = pickle.load(handle)

with open('log_model.pkl', 'rb') as file:
    model = pickle.load(file)

def interpret_result(predicted_result):
    match predicted_result:
        case 0:
            emotion = 'Sad'
            st.write("You are feeling sad")
        case 1:
            emotion = 'Happy'
            st.write("You are feeling happy")
        case 2:
            emotion = 'Love'
            st.write("You are feeling love")
        case 3:
            emotion = 'Angry'
            st.write("You are feeling angry")
        case 4:
            emotion = 'Scared'
            st.write("You are feeling scared")
        case 5:
            emotion = 'Surprise'
            st.write("You are feeling surprised")

    # Call the function to get the image based on the emotion
    image = get_images(emotion)
    if image:
        st.image(image, caption=f"{emotion} Emotion Image", use_column_width=True)

def get_images(emotion):
    # Here you might generate an image or get an image URL
    # For example, if using DALL-E for image generation:
    response = openai.Image.create(
        prompt=f"A cartoon character expressing {emotion}",
        n=1,
        size="256x256"
    )

    image_url = response['data'][0]['url']
    response = requests.get(image_url)

    if response.status_code == 200:
        return Image.open(response.raw)
    else:
        st.error("Failed to retrieve image.")
        return None

st.title("Text Emotion detector: ")

user_text = st.text_input("Write down your emotion here.")

if st.button("Submit"):
    if user_text:
        processed_input = tfidf_vectorizer.transform([user_text])
        predicted_result = model.predict(processed_input)[0]
        # st.write(f"Predicted emotion: {predicted_result}")
        interpret_result(predicted_result)
    else:
        st.write("Write your emotion here :) ")
