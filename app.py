from select import select
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity
import movieposters as mp


data_path = "data/"
links = pd.read_csv(data_path + 'links.csv')
movies = pd.read_csv(data_path + 'movies.csv')
ratings = pd.read_csv(data_path + 'ratings.csv')
tags = pd.read_csv(data_path + 'tags.csv')

# Flags
error_flag = False

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
 

def show_item(recommendations):
    for i in recommendations:
        error_flag = False
        st.text(i)
        try: link = mp.get_poster(title=i)
        except: error_flag = True
        else: st.image(link, width=100)
        if error_flag:
            st.warning('no image')

# App design


# Theme
primaryColor="#F63366"
backgroundColor="#FFFFFF"
secondaryBackgroundColor="#F0F2F6"
textColor="#262730"
font="sans serif"
st.set_page_config(page_title='WBSFLIX group 4', page_icon="random", layout="wide", initial_sidebar_state="auto", menu_items=None)

st.title("WBSFLIX")







# Create a list of all possible genres
all_genres = movies['genres'].str.split(pat='|')
temp_genres = ['All'] # This is default genre to select all movies
for row in all_genres:
    for i in row:
        temp_genres.append(i)

genres = pd.Series(temp_genres)
genres.drop_duplicates(inplace=True)

# Create a list of movie titles
movie_titles = movies['title'].tolist()

# inputs

pop_thres = 20 # predefined
options = ['Show me all time popular movies', 'Show me similar movies to one I like', 'Show me my personalized recommendations']

with st.sidebar:
    st.header('Welcome to WBSFLIX, the world class movie recommender 😉')

    st.image('popcorn_transparent_v2.png', caption=None, width=300, clamp=False, channels="RGB", output_format="auto")
     
    st.header('How may I assist you today?')
    selected_option = st.radio('Please select one', options=options)
    # user_id = st.number_input("User ID", value=1, min_value=1, step=1, format='%i')
    # movie_id = st.number_input("Movie ID", value=1, min_value=1, step=1, format='%i')
    # num_movies = st.number_input("Number of recommendations", value=1, min_value=1, step=1, format='%i')
    # genre = st.selectbox('Genre', options=genres)

    

if selected_option == options[0]:
    st.header('Are you in a mood for a particular genre? Select the genre and let us know how many recommendations you would like')
    
    # Inputs
    genre = st.selectbox('Genre', options=genres)
    num_movies = st.slider("Number of recommendations", min_value=1, max_value=20, value=5, step=1, format=None, key='n-movies', disabled=False)
    #num_movies = st.number_input("Number of recommendations", value=5, min_value=1, step=1, format='%i')

    # get recommendations
    recommendations = get_popular_movies(pop_thres, num_movies, genre)

    # print the list
    show_item(recommendations)
    #for i in recommendations:
    #    st.text(i)

elif selected_option == options[1]:
    st.header('Do you like a particular movie? Tell us which and we will recommend similar movies')

    # Inputs
    selected_movie = st.selectbox('Movie', options=movie_titles)
    movie_id = movies[movies['title'].str.find(selected_movie) != -1]['movieId'].values[0]
    #movie_id = st.number_input("Movie ID", value=1, min_value=1, step=1, format='%i')
    num_movies = st.slider("Number of recommendations", min_value=1, max_value=20, value=5, step=1, format=None, key='n-movies', disabled=False)
    #num_movies = st.number_input("Number of recommendations", value=5, min_value=1, step=1, format='%i')

    # get recommendations
    try:
        recommendations = get_similar_movies(movie_id, num_movies)
        # print the list
        show_item(recommendations)
        #for i in recommendations:
        #    st.text(i)
    except:
        error_flag = True

    if error_flag:
        st.warning('Oops! Selected movie has no reviews yet. Please try another movie!')


elif selected_option == options[2]:
    st.header('Tell us your UserId and we will show you your personalized recommendations')

    # Inputs
    user_id = st.number_input("User ID", value=1, min_value=1, step=1, format='%i')
    num_movies = st.slider("Number of recommendations", min_value=1, max_value=20, value=5, step=1, format=None, key='n-movies', disabled=False)
    #num_movies = st.number_input("Number of recommendations", value=5, min_value=1, step=1, format='%i')


    # get recommendations
    recommendations = weighted_user_rec(user_id, num_movies)
    
    # print the list
    show_item(recommendations)
    #for i in recommendations:
    #    st.text(i)
     


st.header('Have fun watching!')
st.image('Minion_movie.gif', caption=None, width=700, clamp=False, channels="RGB", output_format="auto")


#st.image(mp.get_poster(id=114709), width=50)
#st.balloons()