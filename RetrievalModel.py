"""Retrieval Model using TF Version 2.

Based on the code from this page: https://www.tensorflow.org/recommenders/examples/basic_retrieval
"""

import logging
import os
import pprint
import tempfile

from typing import Dict, Text

import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow_recommenders as tfrs

def get_ratings_data():
    ratings_original = pd.read_csv('DataFiles/ratings_data_0.csv').append(pd.read_csv('DataFiles/ratings_data_1.csv'))
    ratings_original = ratings_original[['ratings', 'user_idx', 'game_idx']]
    ratings_original.rename(columns={'ratings': 'rating'}, inplace=True)
    value_dict = {name: values for name, values in ratings_original.items()}
    value_dict['rating'] = np.asarray(value_dict['rating'].astype('float32'))
    value_dict['user_idx'] = np.asarray(value_dict['user_idx'].astype('string'))
    value_dict['game_idx'] = np.asarray(value_dict['game_idx'].astype('string'))

    ratings = tf.data.Dataset.from_tensor_slices(value_dict)
    return ratings.shuffle(4_000_000, seed=0, reshuffle_each_iteration=False)

def get_games_data():
    games_raw = pd.read_csv('DataFiles/games.csv')
    games_small = games_raw[['game_idx', 'year', 'rating']]
    # value_dict = {name: values for name, values in games_small.items()}
    game_idx_vals = tf.data.Dataset.from_tensor_slices(list(games_raw['game_idx'].values))
    year_vals = tf.data.Dataset.from_tensor_slices(games_raw['year'].values)
    rating_vals = tf.data.Dataset.from_tensor_slices(games_raw['rating'].values)
    data = tf.data.Dataset.zip((game_idx_vals, year_vals, rating_vals))

    return data

if __name__ == '__main__':
    # tf.autograph.set_verbosity(100)
    tf.get_logger().setLevel(logging.ERROR)
    tf.random.set_seed(0)

    ratings = get_ratings_data()
    train = ratings.take(3_000_000)
    test = ratings.skip(3_000_000).take(4_000_000)
    movie_titles = ratings.batch(1_000_000).map(lambda x: x["game_idx"])
    user_ids = ratings.batch(1_000_000).map(lambda x: x["user_idx"])

    unique_movie_titles = np.unique(np.concatenate(list(movie_titles)))
    unique_user_ids = np.unique(np.concatenate(list(user_ids)))

    embedding_dimension = 10

    user_model = tf.keras.Sequential([
        tf.keras.layers.StringLookup(
            vocabulary=unique_user_ids, mask_token=None),
        # We add an additional embedding to account for unknown tokens.
        tf.keras.layers.Embedding(len(unique_user_ids) + 1, embedding_dimension)])

    movie_model = tf.keras.Sequential([
        tf.keras.layers.StringLookup(
            vocabulary=unique_movie_titles, mask_token=None),
        tf.keras.layers.Embedding(len(unique_movie_titles) + 1, embedding_dimension)])

    games = get_games_data()

    metrics = tfrs.metrics.FactorizedTopK(
        candidates=games.batch(128).map(movie_model))

    print("testing")