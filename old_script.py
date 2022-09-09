import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity
#import movieposters as mp
st.write("Streamlit version:", st.__version__)


data_path = "C:/Users/ppadalka/OneDrive - NI/Documents/My Docs/git/wbs_flix_project/data/"
links = pd.read_csv(data_path + 'links.csv')
movies = pd.read_csv(data_path + 'movies.csv')
ratings = pd.read_csv(data_path + 'ratings.csv')
tags = pd.read_csv(data_path + 'tags.csv')


# Function definitions
# popularity-based (n, pop_thres)
def get_popular_movies(pop_threshold, num_movies, genre='All'):
    if (genre != 'All'):
        # Sort movies with selected genre
        movies_with_genres = ratings.merge(movies, left_on='movieId', right_on="movieId")
        selected_genre_movies = movies_with_genres[movies_with_genres['genres'].str.find(genre)!=-1]
    else:
        selected_genre_movies = ratings
    
    # Create a df with avg rating and number of ratings for each movie
    ratings_df = pd.DataFrame(selected_genre_movies.groupby('movieId')['rating'].mean())
    ratings_df['rating_count'] = selected_genre_movies.groupby('movieId')['rating'].count()

    # Select top num_movies based on popularity threshold
    recommended_movies = ratings_df[ratings_df['rating_count'] >= pop_threshold].sort_values('rating', ascending=False).head(num_movies)
    recommended_movies['name'] = recommended_movies.index.to_series().map(lambda x: movies[movies['movieId']==x].title.values[0])

    return recommended_movies['name'].tolist()


# item-based (n, item_id)
def get_similar_movies(movie_id, num_movies):
    # Create user-item matrix
    user_item_matrix = pd.pivot_table(data=ratings, values='rating', index='userId', columns='movieId')

    # Collect ratings for selected movie
    movie_ratings = user_item_matrix[movie_id]

    # Correlation of desired movie with other movies
    movie_corr = user_item_matrix.corrwith(movie_ratings)

    corr_df = pd.DataFrame(movie_corr, columns=['PearsonR'])
    corr_df.dropna(inplace=True)

    # Create a df with avg. ratings and rating_count
    rating = pd.DataFrame(ratings.groupby('movieId')['rating'].mean())
    rating['rating_count'] = ratings.groupby('movieId')['rating'].count()

    # Join corr_df with rating to get correlation and popularity
    corr_summary = corr_df.join(rating['rating_count'])
    corr_summary.drop(movie_id, inplace=True) # drop desired rest_id itself

    # Select only movies with more than 10 reviews
    recommendation = corr_summary[corr_summary['rating_count']>=10].sort_values('PearsonR', ascending=False).head(num_movies)

    # Create df with placeID and name
    movie_names =  movies[['movieId', 'title']]

    recommendation = recommendation.merge(movie_names, left_index=True, right_on="movieId")

    return recommendation['title'].tolist()



# user-based (n, user_id)
def weighted_user_rec(user_id, num_movies):
    # Create user-item matrix
    user_item_matrix = pd.pivot_table(data=ratings, values='rating', index='userId', columns='movieId')

    # Fill NAs with 0
    user_item_matrix.fillna(0, inplace=True)

    # Compute cosine similarities
    user_similarities = pd.DataFrame(cosine_similarity(user_item_matrix), columns=user_item_matrix.index, index=user_item_matrix.index)

    # Compute the weights for desired user
    weights = (user_similarities.query("userId!=@user_id")[user_id] / sum(user_similarities.query("userId!=@user_id")[user_id]))

    # Select movies that have not been rated by the user
    not_rated_movies = user_item_matrix.loc[user_item_matrix.index!=user_id, user_item_matrix.loc[user_id,:]==0]

    # Dot product (multiplication) of the not-rated-movies and the weights
    weighted_averages = pd.DataFrame(not_rated_movies.T.dot(weights), columns=["predicted_rating"])

    # Create df with movieId and name
    movie_names =  movies[['movieId', 'title']]

    recommendations = weighted_averages.merge(movie_names, left_index=True, right_on="movieId")

    top_recommendations = recommendations.sort_values("predicted_rating", ascending=False).head(num_movies)

    return top_recommendations['title'].tolist()



# App design

st.title("Welcome to WBSFLIX")

# Create a list of all possible genres
all_genres = movies['genres'].str.split(pat='|')
temp_genres = ['All'] # This is default genre to select all movies
for row in all_genres:
    for i in row:
        temp_genres.append(i)

genres = pd.Series(temp_genres)
genres.drop_duplicates(inplace=True)


# inputs

pop_thres = 20 # predefined

with st.sidebar:
    user_id = st.number_input("User ID", value=1, min_value=1, step=1, format='%i')
    movie_id = st.number_input("Movie ID", value=1, min_value=1, step=1, format='%i')
    num_movies = st.number_input("Number of recommendations", value=1, min_value=1, step=1, format='%i')
    genre = st.selectbox('Genre', options=genres)


tab_popularity, tab_item, tab_user = st.tabs(["Popular Movies", "Item Based", "User Based"])

with tab_popularity:
    st.header("Here are your popular movie recommendations...")
    # get recommendations
    recommendations = get_popular_movies(pop_thres, num_movies, genre)

    # print the list
    for i in recommendations:
        st.text(i)


with tab_item:
    st.header("Liked a particular movie? Here are some more similar movies...")

    # get recommendations
    recommendations = get_similar_movies(movie_id, num_movies)

    # print the list
    for i in recommendations:
        st.text(i)


with tab_user:
    st.header("You may like these movies based on your interests...")

    # get recommendations
    recommendations = weighted_user_rec(user_id, num_movies)

    # print the list
    for i in recommendations:
        st.text(i)



#st.image(mp.get_poster(id=114709), width=50)
st.balloons()