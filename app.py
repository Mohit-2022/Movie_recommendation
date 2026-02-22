import streamlit as st
import pickle
import pandas as pd
import requests

st.markdown("""
<style>
body {
    background-color: #0e1117;
}
.big-font {
    font-size:50px !important;
    color:#ff4b4b;
    text-align:center;
}
</style>
""", unsafe_allow_html=True)

# Load Data
movies_dict = pickle.load(open('movies_dict.pkl','rb'))
movies = pd.DataFrame(movies_dict)

similarity = pickle.load(open('similarity.pkl','rb'))

def fetch_poster(movie_title):

    url = f"https://api.themoviedb.org/3/search/movie?api_key=d24e81c9c86b61278c12d125150ded6d&query={movie_title}"
    
    data = requests.get(url).json()
    
    if len(data['results']) == 0:
        return "https://via.placeholder.com/300x450?text=No+Image"
    
    poster_path = data['results'][0]['poster_path']
    
    full_path = "https://image.tmdb.org/t/p/w500/" + poster_path
    
    return full_path

# Recommendation Function
def recommend(movie):

    movie_index = movies[movies['title'] == movie].index[0]
    distances = similarity[movie_index]
    
    movie_list = sorted(list(enumerate(distances)),reverse=True,key=lambda x:x[1])[1:6]

    recommended_movies = []
    recommended_posters = []

    for i in movie_list:
        movie_id=movies.iloc[i[0]].movie_id
        recommended_movies.append(movies.iloc[i[0]].title)
        recommended_posters.append(fetch_poster(movie_id))

    return recommended_movies, recommended_posters


# Streamlit UI
st.markdown('<p class="big-font">🎬 Movie Recommendation System</p>', unsafe_allow_html=True)

selected_movie_name = st.selectbox(
    'Select a movie',
    movies['title'].values
)

if st.button('Recommend'):

    names, posters = recommend(selected_movie_name)

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.image(posters[0])
        st.text(names[0])

    with col2:
        st.image(posters[1])
        st.text(names[1])

    with col3:
        st.image(posters[2])
        st.text(names[2])

    with col4:
        st.image(posters[3])
        st.text(names[3])

    with col5:
        st.image(posters[4])
        st.text(names[4])
