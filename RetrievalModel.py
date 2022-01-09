"""Retrieval Model using TF Version 2.

Based on the code from this page: https://www.tensorflow.org/recommenders/examples/basic_retrieval

Some code will contain "movie" terminology to match Google's code. This will be changed later.

Work in progress. Model does not currently run
"""

import logging
from typing import Dict, Text

import os
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_recommenders as tfrs

PATH = os.getcwd() + '\\model'
EPOCHS = 3
DIMENSIONS = 12


def get_ratings_data():
    ratings_original = pd.read_csv('DataFiles/ratings_data_0.csv').append(pd.read_csv('DataFiles/ratings_data_1.csv'))
    ratings_original = ratings_original[['rating_int', 'user_idx', 'game_idx']]
    title_join = pd.read_csv('DataFiles/games.csv')
    title_join = title_join[['game_idx', 'title']]
    title_join.drop_duplicates(inplace=True)
    ratings_original = ratings_original.merge(title_join, on='game_idx')
    ratings_original.rename(columns={'rating_int': 'rating'}, inplace=True)
    value_dict = {name: values for name, values in ratings_original.items()}
    value_dict['rating'] = np.asarray(value_dict['rating'].astype('float32'))
    value_dict['user_idx'] = np.asarray(value_dict['user_idx'].astype('string'))
    value_dict['game_idx'] = np.asarray(value_dict['game_idx'].astype('string'))
    ratings_tf = tf.data.Dataset.from_tensor_slices(value_dict)
    return ratings_tf.shuffle(4_000_000, seed=0, reshuffle_each_iteration=False)


def get_games_data():
    games_raw = pd.read_csv('DataFiles/games.csv')
    games_raw = games_raw[['title', 'year', 'rating']]
    games_raw['title'] = games_raw['title'].astype('string')
    games_raw['year'] = games_raw['year'].astype('int32')
    games_raw['rating'] = games_raw['rating'].astype('int32')
    # value_dict = {name: values for name, values in games_small.items()}
    game_idx_vals = tf.data.Dataset.from_tensor_slices(list(games_raw['title'].values))

    '''
    Commented out. Side features may be used in more advanced model
    year_vals = tf.data.Dataset.from_tensor_slices(list(games_raw['year'].values))
    rating_vals = tf.data.Dataset.from_tensor_slices(list(games_raw['rating'].values))
    data = tf.data.Dataset.zip((game_idx_vals, year_vals, rating_vals))
    '''

    return game_idx_vals


class BGGRetrievalModel(tfrs.Model):

    def __init__(self, user_model_retrieval, movie_model_retrieval):
        super().__init__()
        self.movie_model: tf.keras.Model = movie_model_retrieval
        self.user_model: tf.keras.Model = user_model_retrieval
        self.task: tf.keras.layers.Layer = task

    def compute_loss(self, features: Dict[Text, tf.Tensor], training=False) -> tf.Tensor:
        # We pick out the user features and pass them into the user model.
        user_embeddings = self.user_model(features["user_idx"])
        # And pick out the movie features and pass them into the movie model,
        # getting embeddings back.
        positive_movie_embeddings = self.movie_model(features["title"])

        # The task computes the loss and the metrics.
        return self.task(user_embeddings, positive_movie_embeddings)


if __name__ == '__main__':
    tf.get_logger().setLevel(logging.ERROR)
    tf.random.set_seed(0)

    ratings = get_ratings_data()
    train = ratings.take(3_000_000)
    test = ratings.skip(3_000_000).take(800_000)
    movie_titles = ratings.batch(300).map(lambda x: x["title"])
    user_ids = ratings.batch(1_000_000).map(lambda x: x["user_idx"])

    unique_movie_titles = np.unique(np.concatenate(list(movie_titles)))
    unique_user_ids = np.unique(np.concatenate(list(user_ids)))

    embedding_dimension = DIMENSIONS

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

    task = tfrs.tasks.Retrieval(metrics=metrics)

    model = BGGRetrievalModel(user_model, movie_model)
    model.compile(optimizer=tf.keras.optimizers.Adagrad(learning_rate=0.1))

    cached_train = train.shuffle(1_000_000).batch(5_000).cache()
    cached_test = test.batch(4096).cache()

    model.fit(cached_train, epochs=EPOCHS)

    return_dict = model.evaluate(cached_test, return_dict=True)
    print(return_dict)

    # Create a model that takes in raw query features, and
    index = tfrs.layers.factorized_top_k.BruteForce(model.user_model)
    # recommends movies out of the entire movies dataset.
    index.index_from_dataset(
        tf.data.Dataset.zip((games.batch(100), games.batch(100).map(model.movie_model)))
    )

    # Save the index.
    tf.saved_model.save(index, PATH, options=tf.saved_model.SaveOptions())
