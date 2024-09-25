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
    st.subheader("Step 1: Input Your Text")
    st.write("Enter the text for which you want to analyze the emotion in the text box provided.")

    st.subheader("Step 2: View the Results")

with st.expander("More Information"):
    st.write(display)

user_text = st.text_input("Write down your emotion here.")

if st.button("Submit"):
    if user_text:
        processed_input = tfidf_vectorizer.transform([user_text])
        predicted_result = model.predict(processed_input)[0]
        interpret_result(predicted_result)
    else:
        st.write("Write your emotion here :) ")
