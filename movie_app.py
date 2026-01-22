import pickle
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity
import difflib #to compare texts
import requests

def get_movie_details(movie_name):
    API_KEY= "3514848a73231cc7557bdd3059b759b4"
    search_url = f"https://api.themoviedb.org/3/search/movie?api_key={API_KEY}&query={movie_name}"
    try:
        response= requests.get(search_url) #sends your request to TMDb
        data=response.json() #TMDb sends back data in JSON format
        if data['results']:
            movie=data['results'][0] # Get the first result
            poster_path=movie.get('poster_path') # Get poster file path
            if poster_path:
                poster_url = f"https://image.tmdb.org/t/p/w500{poster_path}"
                return {
                    'poster':poster_url,
                    'title':movie['title'],
                    'year':movie.get('release_date','')[:4],
                    'rating':movie.get('vote_average','N/A')
                    }
        return None
    except:
        return None


# Load model and vectorizer
df = pickle.load(open("movies_df.sav", "rb"))
vectorizer = pickle.load(open("tfidf_vectorizer.sav", "rb"))
feature_vectors = pickle.load(open("tfidf_matrix.sav", "rb"))

st.title("Movie recommendation system")

#input message
user_message=st.text_area("enter your fav movie")
n = top_n = st.slider("How many recommendations?", 1, 20, 10)


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

        count=1
        for movie in sorted_similar_movies:
            movie_index = movie[0]
            similarity_index = movie[1]

            if movie_index == index_movie:
                continue
            title = df[df['index'] == movie_index]['title'].values[0]
            st.write(f"{count}. {title} - similarity :{similarity_index*100: .1f}%")
            poster = get_movie_details(title)
    
            if poster:
                st.image(poster['poster'], width=300)           
                st.write(f"**{poster['title']}**")              
                st.caption(f"⭐ {poster['rating']} | {poster['year']}")  
            else:
                st.error("Poster not found!")
            count += 1
            
            if count > n:
                break
