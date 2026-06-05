import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.feature_extraction import text
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(
    page_title="Movie Matcher",
    page_icon="🎬",
    layout="wide"
)

st.markdown("""
    <style>
    .main .block-container {padding-top: 2rem; padding-bottom: 2rem;}
    h1, h2, h3 {font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;}
    </style>
""", unsafe_allow_html=True)


@st.cache_data(show_spinner="Setting up movie catalog dataset...")
def load_data():
    df = pd.read_csv('movies.csv')
    df['display_title'] = df['title'] + " (" + df['genres'].str.replace('|', ', ', regex=False) + ")"
    df['genres_clean'] = df['genres'].str.replace('|', ' ', regex=False)
    df['search_metadata'] = df['title'] + " " + df['genres_clean']

    df['genre_set'] = df['genres'].apply(lambda x: set(x.split('|')) if x != '(no genres listed)' else set())
    return df


@st.cache_resource(show_spinner="Training core recommendation engines...")
def train_recommendation_engines(df):
    base_stopwords = text.ENGLISH_STOP_WORDS
    cinematic_noise = {
        'movie', 'movies', 'film', 'films', 'part', 'ii', 'iii', 'v', 'version',
        'edition', 'listed', 'genres', 'no', 'story', 'stories'
    }
    final_stopwords = list(base_stopwords.union(cinematic_noise))

    tfidf = TfidfVectorizer(stop_words=final_stopwords, min_df=1, token_pattern=r'(?u)\b\w+\b')
    tfidf_matrix = tfidf.fit_transform(df['search_metadata'])

    count_vec = CountVectorizer(tokenizer=lambda x: x.split('|'), token_pattern=None, lowercase=False)
    count_matrix = count_vec.fit_transform(df['genres'])

    return tfidf_matrix, count_matrix, final_stopwords


df = load_data()
tfidf_matrix, count_matrix, used_stopwords = train_recommendation_engines(df)


def calculate_overlap_score(target_set, recommendation_set):
    """Calculates how closely two lists of genres match mathematically (Jaccard Index)"""
    if not target_set or not recommendation_set:
        return 0.0
    intersection = len(target_set.intersection(recommendation_set))
    union = len(target_set.union(recommendation_set))
    return float(intersection) / union


def generate_top_recommendations(movie_index, number_of_suggestions=10):
    query_vector_1 = tfidf_matrix[movie_index]
    similarity_scores_1 = cosine_similarity(query_vector_1, tfidf_matrix).flatten()
    best_matches_1 = np.argsort(similarity_scores_1)[-(number_of_suggestions + 1):-1][::-1]

    query_vector_2 = count_matrix[movie_index]
    similarity_scores_2 = cosine_similarity(query_vector_2, count_matrix).flatten()
    best_matches_2 = np.argsort(similarity_scores_2)[-(number_of_suggestions + 1):-1][::-1]

    return best_matches_1, similarity_scores_1[best_matches_1], best_matches_2, similarity_scores_2[best_matches_2]


st.sidebar.title("🎬 Movie Matcher Controls")
st.sidebar.write("Find and customize your movie recommendations below.")
st.sidebar.markdown("---")

user_search = st.sidebar.text_input("1. Type a movie title keyword:", value="Naruto")

matching_movies = df[df['title'].str.contains(user_search, case=False, na=False)]

if not matching_movies.empty:
    selected_display = st.sidebar.selectbox(
        "2. Select your exact movie:",
        options=matching_movies['display_title'].tolist()
    )

    chosen_index = df[df['display_title'] == selected_display].index[0]
    selected_movie = df.iloc[chosen_index]

    st.sidebar.markdown("---")
    total_suggestions = st.sidebar.slider("3. How many suggestions do you want?", min_value=5, max_value=20, value=10)

    st.title("🍿 Simple Movie Recommendation Dashboard")
    st.markdown("Discover similar films using two different smart matching methods. " 
                "This intelligent recommendation application serves as a machine learning sandbox designed to predict and analyze movie similarity profiles. By leveraging natural language processing techniques, the system transforms raw textual attributes into mathematical feature vectors. Users can select a target movie to compare two distinct content-based algorithmic approaches side-by-side in real-time. The platform automatically runs statistical evaluation scores to track categorical accuracy and detect anomalies within the dataset. Ultimately, this interactive dashboard acts as a comprehensive proof-of-concept demonstrating how discrete data models solve complex content discovery problems.")

    st.subheader("🎯 Your Reference Movie Choice")
    st.info(f"**Selected Title:** {selected_movie['title']} \n\n **Categorized Genres:** {selected_movie['genres']}")

    top_indices_1, scores_1, top_indices_2, scores_2 = generate_top_recommendations(chosen_index,
                                                                            number_of_suggestions=total_suggestions)

    rec_A_df = df.iloc[top_indices_1].copy()
    rec_A_df['Similarity Score'] = scores_1

    rec_B_df = df.iloc[top_indices_2].copy()
    rec_B_df['Similarity Score'] = scores_2

    target_g_set = selected_movie['genre_set']
    rec_A_df['Genre Accuracy'] = rec_A_df['genre_set'].apply(lambda x: calculate_overlap_score(target_g_set, x))
    rec_B_df['Genre Accuracy'] = rec_B_df['genre_set'].apply(lambda x: calculate_overlap_score(target_g_set, x))

    mean_accuracy_A = rec_A_df['Genre Accuracy'].mean()
    mean_accuracy_B = rec_B_df['Genre Accuracy'].mean()

    st.markdown("---")
    st.subheader("Your Personalized Recommendations")

    column_left, column_right = st.columns(2)

    with column_left:
        st.markdown("### Similarity Mix")
        st.caption("Matches movies based on a mix of similar title words, franchises, and genres.")

        # Isolate relevant recommendations data
        recommendations_A = df.iloc[top_indices_1][['title', 'genres']].reset_index(drop=True)
        recommendations_A.index += 1  # Make list start at 1 instead of 0
        st.dataframe(recommendations_A, use_container_width=True)

    with column_right:
        st.markdown("### Strict Genre Match")
        st.caption("Matches movies strictly based on sharing the exact same genre categories.")

        recommendations_B = df.iloc[top_indices_2][['title', 'genres']].reset_index(drop=True)
        recommendations_B.index += 1  # Make list start at 1 instead of 0
        st.dataframe(recommendations_B, use_container_width=True)

    st.markdown("---")
    st.subheader("📊 Quantitative Performance Evaluation")
    st.markdown("We track how accurately each algorithm matches the movie categories chosen by the user.")

    col_m1, col_m2 = st.columns(2)
    with col_m1:
        st.metric(
            label="Similarity Mix Category Accuracy",
            value=f"{mean_accuracy_A * 100:.1f}%",
            delta=f"{(mean_accuracy_A - mean_accuracy_B) * 100:.1f}% vs Approach B"
        )
    with col_m2:
        st.metric(
            label="Strict Genre Category Accuracy",
            value=f"{mean_accuracy_B * 100:.1f}%",
            delta=f"{(mean_accuracy_B - mean_accuracy_A) * 100:.1f}% vs Approach A"
        )


    st.markdown("---")
    st.subheader("Diagnostics & Error Analysis")

    col_e1, col_e2 = st.columns(2)

    with col_e1:
        st.markdown("##### Dataset Edge Cases Encountered")
        if selected_movie['genres'] == '(no genres listed)':
            st.error(
                "⚠️ **Current Selection Triggered a Missing Data Profile Exception!** This specific movie contains zero assigned genre attributes. Approach B is completely blind here; Approach A can still generate suggestions by falling back entirely on title character matching.")
        else:
            st.success(
                "✅ **Current Target Selection Status:** Active. Genre vectors contain clean, fully populated classifications.")

        st.markdown("""
        * **The Unordered Tie-Breaker Problem:** Because Approach B matches strict categorical tags (like `Drama`), dozens of movies can achieve a matching score of `1.000` simultaneously. Without extra variables, the sorting algorithm relies on the raw index sequence of the dataset file.
        """)

    with col_e2:
        st.markdown("##### Future Engineering Upgrades")
        st.markdown("""
        To elevate this baseline system to an enterprise-grade recommendation layout, consider integrating these advanced components:

        1. **System Metadata Synthesis:** Append external datasets (such as IMDB or TMDB plot overview descriptions) into the text arrays to expand the matching vocabulary.
        2. **Weighted Score Blending ($\alpha$):** Instead of keeping the approaches completely separate, combine both scoring systems into a single balanced equation:
        """)
        st.latex(r"\text{Final Score} = \alpha \cdot \text{Score}_A + (1 - \alpha) \cdot \text{Score}_B")

else:

    st.title("🍿 Simple Movie Recommendation Dashboard")
    st.sidebar.error("❌ No titles match your search query.")
    st.warning("Please adjust your search keyword in the sidebar panel to find matching movies in the database.")
