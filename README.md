# Emotion Recognition using Logistic Regression

## Overview
This project uses logistic regression to classify and interpret the current emotions of users based on their input data. It provides an interactive user interface that displays the predicted emotion along with a corresponding message and image.

## Features
- Predicts user emotions: Sad, Happy, Love, Angry, Scared, Surprise
- Displays a message and an image representing the predicted emotion
- User-friendly interface built with Streamlit

## Installation
To get started, clone this repository and install the required packages:

# bash
git clone https://github.com/jasonchiahs/Emotion-Text-Detection.git
cd emotion-recognition
pip install -r requirements.txt

Requirements
Python 3.x
Streamlit
scikit-learn
numpy
pandas
OpenAI
(Other dependencies as needed)

## Usage
Run the application:
terminal: streamlit run app.py
Follow the instructions on the web interface to input data and receive emotion predictions.

# Model Training
The logistic regression model was trained on Kaggle: Emotion dataset to predict emotions based on 'text' & 'label'.

# Contributing
Contributions are welcome! Feel free to submit a pull request or open an issue to discuss improvements.

# Room for improvement

Being more specific can definitely enhance the clarity and impact of your message. Here are a few suggestions on how to add more detail:

Explain the Dataset Issues: Describe the specific problems with the dataset. For example, mention if there are certain phrases that are frequently misinterpreted or if there are inconsistencies in labeling.

Impact on Model Performance: Specify how these dataset issues affected the model. Did it lead to poor accuracy, or did it confuse the model between certain emotions?

Future Improvements: Consider outlining specific steps you plan to take for improvement. For example, mention data cleaning techniques, exploring different models, or gathering a more comprehensive dataset.

Revised Example
Here’s a more detailed version incorporating these elements:

There is still significant room for improvement in this project. The dataset used is not as clean as anticipated; it contains misinterpreted meanings for phrases such as "I feel blue" (often understood as sadness) that can confuse the model. These inconsistencies have led to reduced accuracy in emotion classification. This project serves as a showcase of my ability to implement NLP models using both traditional machine learning and deep learning techniques. Moving forward, I aim to clean the dataset further, explore alternative models, and incorporate more diverse data to enhance performance.

Moving forward, I aim to clean the dataset further, explore alternative models, and incorporate more diverse data to enhance performance. My ultimate goal is to secure a position as a machine learning engineer in the near future.
