import re
import pandas as pd
import numpy as np

def preprocessing_text(text):
    text = text.lower()
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'\b(?:a|href|http|https|www)\b', '', text)
    text = text.strip()
    return text

def emotion_to_features(percentages):
    mapping = {
        'Sadness': {'danceability': 0.3, 'energy': 0.2, 'loudness': -10, 'speechiness': 0.5, 'acousticness': 0.8, 'instrumentalness': 0.1, 'liveness': 0.4, 'valence': 0.2},
        'Joy': {'danceability': 0.8, 'energy': 0.9, 'loudness': -5, 'speechiness': 0.1, 'acousticness': 0.2, 'instrumentalness': 0.0, 'liveness': 0.8, 'valence': 0.9},
        'Love': {'danceability': 0.5, 'energy': 0.5, 'loudness': -6, 'speechiness': 0.1, 'acousticness': 0.5, 'instrumentalness': 0.0, 'liveness': 0.5, 'valence': 0.7},
        'Anger': {'danceability': 0.5, 'energy': 0.8, 'loudness': -2, 'speechiness': 0.1, 'acousticness': 0.2, 'instrumentalness': 0.0, 'liveness': 0.5, 'valence': 0.3},
        'Fear': {'danceability': 0.2, 'energy': 0.4, 'loudness': -8, 'speechiness': 0.4, 'acousticness': 0.5, 'instrumentalness': 0.1, 'liveness': 0.3, 'valence': 0.1},
        'Surprise': {'danceability': 0.5, 'energy': 0.7, 'loudness': -4, 'speechiness': 0.3, 'acousticness': 0.2, 'instrumentalness': 0.0, 'liveness': 0.5, 'valence': 0.6},
    }
    # labelled_rawnum =  {label_mapping[k]:v for k,v in raw_num.items()}
    features = {key: 0 for key in next(iter(mapping.values())).keys()}
    # percentages = labelled_rawnum * 100
    total_percentage = sum(percentages.values())

    for emotion, percent in percentages.items():
        if emotion in mapping:
            weight = percent / total_percentage
            for feature, value in mapping[emotion].items():
                features[feature] += value * weight

    return features

def map_emotionresults(log_model_result):
    label_mapping = {
        0: 'Sadness',
        1: 'Joy',
        2: 'Love',
        3: 'Anger',
        4: 'Fear',
        5: 'Surprise'
    }
    labeled_probabilities = {label_mapping[k]:v*100 for k,v in log_model_result.items()}
    return labeled_probabilities
