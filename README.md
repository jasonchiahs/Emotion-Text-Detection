# Emotion Recognition using Logistic Regression

## Overview
Emotion Text Analysis is an innovative project designed to uncover and understand the emotional nuances in textual data. By leveraging natural language processing (NLP) and machine learning techniques, this project automatically identifies and categorizes emotions in written content, such as social media posts, reviews, and articles, and recommends songs based on the analyzed mood.

## Features
# Emotion Identification:
- Predicts user emotions: Sad, Happy, Love, Angry, Scared, Surprise
- Trained model using logistic regression
- Optimized parameters through grid search
- Achieved 92% accuracy using TF-IDF vectorization

# Song Selection:
- Utilizes K-means clustering to find nearest clusters based on attributes
- A dedicated function converts these emotions into song attributes, fitting them into the K-means model and using Euclidean distance to find the nearest recommended songs
- Song recommendations are provided via Spotify

## Installation
To get started, clone this repository and install the required packages:

# bash
1. git clone https://github.com/jasonchiahs/Emotion-Text-Detection.git
2. cd emotion-recognition
3. pip install -r requirements.txt

Requirements
- Python 3.x
- Streamlit
- scikit-learn
- numpy
- pandas
- OpenAI

## Usage
1. Run the application:
2. terminal: streamlit run app.py or run app_with_noAIgen.py 
3. Follow the instructions on the web interface to input data and receive emotion predictions.

# Model Training
The logistic regression model was trained on my local machine using the Kaggle Emotion dataset to predict emotions based on the 'text' and 'label' columns.

# Contributing
Contributions are welcome! Feel free to submit a pull request or open an issue to discuss improvements.

# Room for improvement

There is still significant room for improvement in this project. The dataset used is not as clean as anticipated; it contains misinterpreted meanings for phrases such as "I feel blue" (often understood as sadness) that can confuse the model. These inconsistencies have led to reduced accuracy in emotion classification. This project serves as a showcase of my ability to implement NLP models using both traditional machine learning and deep learning techniques. Moving forward, I aim to clean the dataset further, explore alternative models, and incorporate more diverse data to enhance performance.

Moving forward, I aim to clean the dataset further, explore alternative models, and incorporate more diverse data to enhance performance. My ultimate goal is to secure a position as a machine learning engineer in the near future.
