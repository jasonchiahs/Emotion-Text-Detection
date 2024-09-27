import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

label_mapping = {
    0: 'Sadness',
    1: 'Joy',
    2: 'Love',
    3: 'Anger',
    4: 'Fear',
    5: 'Surprise'
}

def plot_pie_chart(predicted_result):
    threshold = 5
    percentage_result = predicted_result * 100
    mask = percentage_result >= threshold
    display_predict = percentage_result[mask]
    display_label = np.array(list(label_mapping.values()))[mask]
    colors = ['#ff9999', '#66b3ff']  # Add more colors if needed

    # Create a pie chart
    plt.figure(figsize=(3, 3))
    explode = [0.1] * len(display_predict)  # Explode all slices for emphasis

    wedges, texts, autotexts = plt.pie(
        display_predict,
        labels=display_label,
        autopct='%1.3f%%',  # Display percentages
        startangle=90,
        explode=explode,
        colors=colors,
        shadow=True  # Add shadow for depth
    )

    # Customize the font size and style
    plt.setp(texts, size=12, weight='bold')
    plt.setp(autotexts, size=10, weight='bold', color='white')

    plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
    plt.title('Predicted Class Probabilities', fontsize=16, weight='bold')
    st.pyplot(plt)
    return
