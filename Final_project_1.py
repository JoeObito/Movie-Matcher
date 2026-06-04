import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(
    page_title="Movie Matcher",
    page_icon="🎬",
    layout="wide"
)

@st.cache_data(show_spinner=False)
def load_data():
    df = pd.read_csv('movies.csv')
    df['genres_clean'] = df['genres'].str.replace('|', ' ', regex=False)
    df['search_metadata'] = df['title'] + " " + df['genres_clean']
    return df


@st.cache_resource(show_spinner=False)
def train_recommendation_engines(df):
    tfidf = TfidfVectorizer(stop_words='english', token_pattern=r'(?u)\b\w+\b')
    tfidf_matrix = tfidf.fit_transform(df['search_metadata'])

    count_vec = CountVectorizer(tokenizer=lambda x: x.split('|'), lowercase=False)
    count_matrix = count_vec.fit_transform(df['genres'])

    return tfidf_matrix, count_matrix

df = load_data()
tfidf_matrix, count_matrix = train_recommendation_engines(df)

def generate_top_recommendations(movie_index, number_of_suggestions=10):
    query_vector_1 = tfidf_matrix[movie_index]
    similarity_scores_1 = cosine_similarity(query_vector_1, tfidf_matrix).flatten()
    best_matches_1 = np.argsort(similarity_scores_1)[-(number_of_suggestions + 1):-1][::-1]

    query_vector_2 = count_matrix[movie_index]
    similarity_scores_2 = cosine_similarity(query_vector_2, count_matrix).flatten()
    best_matches_2 = np.argsort(similarity_scores_2)[-(number_of_suggestions + 1):-1][::-1]

    return best_matches_1, best_matches_2

st.sidebar.title("🎬 Movie Matcher Controls")
st.sidebar.write("Find and customize your movie recommendations below.")
st.sidebar.markdown("---")


user_search = st.sidebar.text_input("1. Type a movie title:", value="Naruto")

matching_movies = df[df['title'].str.contains(user_search, case=False, na=False)]

if not matching_movies.empty:
    formatted_options = matching_movies['title'] + " (" + matching_movies['genres'] + ")"
    selected_display = st.sidebar.selectbox("2. Select your exact movie:", options=formatted_options)

    chosen_index = matching_movies.index[formatted_options.tolist().index(selected_display)]
    selected_movie = df.iloc[chosen_index]
else:
    st.sidebar.error("No titles match your search. Try checking your spelling!")
    st.stop()

st.sidebar.markdown("---")
total_suggestions = st.sidebar.slider("3. How many suggestions do you want?", min_value=5, max_value=20, value=10)

st.title("Movie Recommendation Dashboard")
st.markdown("Discover similar films using two different smart matching methods.")

st.subheader("Your Reference Movie Choice")
st.info(f"**Selected Title:** {selected_movie['title']} \n\n **Categorized Genres:** {selected_movie['genres']}")

top_indices_1, top_indices_2 = generate_top_recommendations(chosen_index, number_of_suggestions=total_suggestions)

st.markdown("---")
st.subheader("Your Personalized Recommendations")

column_left, column_right = st.columns(2)

with column_left:
    st.markdown("### Similarity Mix")
    st.caption("Matches movies based on a mix of similar title words, franchises, and genres.")

    recommendations_A = df.iloc[top_indices_1][['title', 'genres']].reset_index(drop=True)
    recommendations_A.index += 1
    st.dataframe(recommendations_A, use_container_width=True)

with column_right:
    st.markdown("### Strict Genre Match")
    st.caption("Matches movies strictly based on sharing the exact same genre categories.")

    recommendations_B = df.iloc[top_indices_2][['title', 'genres']].reset_index(drop=True)
    recommendations_B.index += 1
    st.dataframe(recommendations_B, use_container_width=True)
