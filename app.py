import streamlit as st
import pickle
import pandas as pd


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


# Recommendation Function
def recommend(movie):
    movie_index = movies[movies['title'] == movie].index[0]
    distances = similarity[movie_index]

    movies_list = sorted(list(enumerate(distances)),
                         reverse=True,
                         key=lambda x: x[1])[1:6]

    recommended_movies = []

    for i in movies_list:
        movie_id = movies.iloc[i[0]].movie_id
        recommended_movies.append(movies.iloc[i[0]].title)

    return recommended_movies

# Streamlit UI
st.markdown('<p class="big-font">🎬 Movie Recommendation System</p>', unsafe_allow_html=True)

selected_movie_name = st.selectbox(
    'Select a movie',
    movies['title'].values
)

if st.button("Recommend"):
    names = recommend(selected_movie_name)

    cols = st.columns(5)

    for idx in range(len(names)):
        with cols[idx]:
            st.text(names[idx])
