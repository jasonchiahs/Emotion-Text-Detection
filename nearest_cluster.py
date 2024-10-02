import pandas as pd
import numpy as np
import pickle
from sklearn.metrics import pairwise_distances

with open('K_means.pkl', 'rb') as file:
    kmeans_model = pickle.load(file)
with open('RBscaling.pickle', 'rb') as handle:
    RBscaling = pickle.load(handle)

clustered_frame = pd.read_csv('clustered.csv')
def recommend_songs(user_input, num_recommendations=5):
    user_vector = np.array([user_input.get('danceability', 0),
                            user_input.get('energy', 0),
                            user_input.get('loudness', 0),
                            user_input.get('speechiness', 0),
                            user_input.get('acousticness', 0),
                            user_input.get('instrumentalness', 0),
                            user_input.get('liveness', 0),
                            user_input.get('valence', 0)])

    user_vector_scaled = RBscaling.transform([user_vector])

    distances = pairwise_distances(user_vector_scaled, kmeans_model.cluster_centers_)
    closest_cluster = np.argmin(distances)
    # Recommend songs from the closest cluster
    recommended_songs = clustered_frame[clustered_frame['Cluster'] == closest_cluster]['uri'].sample(n=num_recommendations).tolist()
    return recommended_songs
