"""Takes retrieval model, and produces a top 10 recommended game alongside the user's top 10 rated games."""

import logging
from typing import Dict, Text

import os
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_recommenders as tfrs
from RetrievalModel import get_user_suggestions, get_ratings_data

PATH = os.getcwd() + '\\retrieval'
EPOCHS = 1
DIMENSIONS = 10
YEAR_BINS = 20
GAMES_BATCH = 20
SAVE_MODEL = True


def truncate_string(game_name):
    if len(game_name) > 25:
        return game_name[:25] + '...'
    else:
        return game_name


def get_ratings_df(user_idx):
    ratings_original = pd.read_csv('DataFiles/ratings_data_0.csv').append(pd.read_csv('DataFiles/ratings_data_1.csv'))
    ratings_original = ratings_original[['rating_int', 'user_idx', 'game_idx']]
    title_join = pd.read_csv('DataFiles/games.csv')
    title_join = title_join[['game_idx', 'title', 'year', 'ratings']]
    title_join.rename(columns={'ratings': 'num_ratings'}, inplace=True)
    title_join.drop_duplicates(inplace=True)
    ratings_original = ratings_original.merge(title_join, on='game_idx')
    ratings_original.rename(columns={'rating_int': 'rating'}, inplace=True)

    return ratings_original.loc[ratings_original['user_idx'] == user_idx,
                                ['title', 'rating']].sort_values('rating', ascending=False)


if __name__ == '__main__':
    NUM = 5
    loaded = tf.saved_model.load(PATH)
    df = get_ratings_df(NUM)
    df['title'] = df['title'].apply(truncate_string)
    print(df.head(10))
    results = get_user_suggestions(str(NUM), loaded)
    results['games'] = results['games'].str.decode('utf-8')
    results['games'] = results['games'].apply(truncate_string)
    print(results)
