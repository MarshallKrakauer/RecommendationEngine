"""Creating Data from Game Ratings."""

from collections import defaultdict

import pandas as pd


def return_negative_one():
    return -1


def create_dictionaries(dataframe):
    user_dict_empty = defaultdict(return_negative_one)
    game_dict_empty = defaultdict(return_negative_one)

    i = 0
    for u in dataframe['user'].unique():
        user_dict_empty[u] = i
        i += 1

    i = 0
    for u in dataframe['game_id'].unique():
        game_dict_empty[u] = i
        i += 1

    return user_dict_empty, game_dict_empty


def get_ratings_data():
    ratings_df = pd.read_csv('player_ratings.csv')
    user_dict, game_dict = create_dictionaries(ratings_df)
    ratings_df['user_idx'] = ratings_df['user'].apply(lambda x: user_dict[x])
    ratings_df['game_idx'] = ratings_df['game_id'].apply(lambda x: game_dict[x])
    ratings_df['rating_int'] = ratings_df['rating'].astype(int)
    return ratings_df
