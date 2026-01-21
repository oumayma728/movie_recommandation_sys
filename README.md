### Movie Recommandation system
this project is a content-based movie recommendation system build using TF_IDF Vectorizer and cosine similarity.
the system recommends movies that are similar to a movie entered by the user.

## how it works:
1. Relevant movie features are selected and combined into text.
2. The text is converted into numerical vectors using TfidfVectorizer.
3. Cosine similarity is used to measure how similar movies are to each other.
4. When a user enters a movie title, difflib is used to find the closes match.
5. Then system retunrs a list of movies that are most similar to the selected one.

## Technologies used:
-Python <br>
-Pandas <br>
-Scikit-learn <br>
-Streamlit <br>
-Pickle <br>
