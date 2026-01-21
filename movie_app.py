import pickle
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity
import difflib #to compare texts

# Load model and vectorizer
df = pickle.load(open("movies_df.sav", "rb"))
vectorizer = pickle.load(open("tfidf_vectorizer.sav", "rb"))
feature_vectors = pickle.load(open("tfidf_matrix.sav", "rb"))

st.title("Movie recommendation system")

#input message
user_message=st.text_area("enter your fav movie")
if st.button("Recommand"):
    #get all the movies
    r_movies= df['title'].tolist()
    #finding the closet match
    match=difflib.get_close_matches(user_message,r_movies)
    if len(match) == 0:
        st.error("Movie not found. please try another title")
    else:
        close_match=match[0]
        st.success(f"Closest match found: {close_match}")

        index_movie=df[df['title'] == close_match]['index'].values[0]
        similarity_scores = cosine_similarity(
        feature_vectors[index_movie],
        feature_vectors
        )[0]
        similarity_list = list(enumerate(similarity_scores))
        sorted_similar_movies = sorted(
        similarity_list,
        key=lambda x: x[1],
        reverse=True
        )
        st.subheader("Recommended movies:")

        count = 1
        for movie in sorted_similar_movies:
            movie_index = movie[0]

            if movie_index == index_movie:
                continue
            title = df[df['index'] == movie_index]['title'].values[0]
            st.write(f"{count}. {title}")
            count += 1
            
            if count > 9:
                break
