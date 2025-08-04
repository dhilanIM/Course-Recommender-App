# 🎓 Course Recommender System

This project is an interactive course recommendation app built with **Streamlit**. It suggests new courses based on the ones a user has previously completed.

## 🧠 Recommendation Models

The system supports two recommendation algorithms:

1. **Course Similarity** – Uses cosine similarity between course vectors to suggest similar courses.
2. **Clustering with PCA** – Applies dimensionality reduction (PCA) and KMeans clustering to group users with similar profiles and recommend courses popular in those clusters.

## 🚀 Features

- User-friendly Streamlit interface for selecting completed courses.
- Real-time model training and recommendation generation.
- Dynamic tuning of model parameters via sidebar widgets.

## 🗂️ Files

- `backend.py`: Core logic for loading data, training models, and generating recommendations.
- `recommender_app.py`: Streamlit-based frontend for user interaction.

## 🌐 Demo

[View  demo](https://course-recommender-app.up.railway.app/)
