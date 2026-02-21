import streamlit as st
import pickle
import pandas as pd

# Load Data
movies_dict = pickle.load(open('movies_dict.pkl','rb'))
movies = pd.DataFrame(movies_dict)

similarity = pickle.load(open('similarity.pkl','rb'))

# Recommendation Function
def recommend(movie):

    if movie not in movies['title'].values:
        return ["Movie not found"]

    movie_index = movies[movies['title'] == movie].index[0]

    similar_movies = similarity[movie_index]

    recommended_movies = []

    for i in similar_movies:
        recommended_movies.append(movies.iloc[i].title)

    return recommended_movies

    for i in movie_list:
        recommended_movies.append(movies.iloc[i[0]].title)

    return recommended_movies


# Streamlit UI
st.title('🎬 Movie Recommendation System')

selected_movie_name = st.selectbox(
    'Select a movie',
    movies['title'].values
)

if st.button('Recommend'):

    recommendations = recommend(selected_movie_name)

    for i in recommendations:
        st.write(i)
