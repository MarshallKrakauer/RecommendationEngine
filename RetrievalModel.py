"""Retrieval Model using TF Version 2.

Based on the code from this page: https://www.tensorflow.org/recommenders/examples/basic_retrieval

Some code will contain "movie" terminology to match Google's code. This will be changed later.

Work in progress. Model does not currently run
"""

import logging
from typing import Dict, Text

import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_recommenders as tfrs
import tempfile
import os


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
    games_raw = games_raw[['game_idx', 'year', 'rating']]
    games_raw['game_idx'] = games_raw['game_idx'].astype('string')
    games_raw['year'] = games_raw['year'].astype('int32')
    games_raw['rating'] = games_raw['rating'].astype('int32')
    print(games_raw.dtypes)
    # value_dict = {name: values for name, values in games_small.items()}
    game_idx_vals = tf.data.Dataset.from_tensor_slices(list(games_raw['game_idx'].values))
    year_vals = tf.data.Dataset.from_tensor_slices(list(games_raw['year'].values))
    rating_vals = tf.data.Dataset.from_tensor_slices(list(games_raw['rating'].values))
    data = tf.data.Dataset.zip((game_idx_vals, year_vals, rating_vals))

    return game_idx_vals


class MovielensModel(tfrs.Model):

    def __init__(self, user_model, movie_model):
        super().__init__()
        self.movie_model: tf.keras.Model = movie_model
        self.user_model: tf.keras.Model = user_model
        self.task: tf.keras.layers.Layer = task

    def compute_loss(self, features: Dict[Text, tf.Tensor], training=False) -> tf.Tensor:
        # We pick out the user features and pass them into the user model.
        user_embeddings = self.user_model(features["user_idx"])
        # And pick out the movie features and pass them into the movie model,
        # getting embeddings back.
        positive_movie_embeddings = self.movie_model(features["game_idx"])

        # The task computes the loss and the metrics.
        return self.task(user_embeddings, positive_movie_embeddings)


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

    task = tfrs.tasks.Retrieval(metrics=metrics)

    model = MovielensModel(user_model, movie_model)
    model.compile(optimizer=tf.keras.optimizers.Adagrad(learning_rate=0.1))

    cached_train = train.shuffle(1_000_000).batch(5_000).cache()
    cached_test = test.batch(4096).cache()

    model.fit(cached_train, epochs=1)

    model.evaluate(cached_test, return_dict=True)

    # Create a model that takes in raw query features, and
    index = tfrs.layers.factorized_top_k.BruteForce(model.user_model)
    # recommends movies out of the entire movies dataset.
    index.index_from_dataset(
        tf.data.Dataset.zip((games.batch(100), games.batch(100).map(model.movie_model)))
    )

    # Get recommendations.
    _, titles = index(tf.constant(["22"]))
    print(f"Recommendations for user 22: {titles[0, :3]}")

    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "model")

        # Save the index.
        tf.saved_model.save(
            index,
            path,
            options=tf.saved_model.SaveOptions(namespace_whitelist=["Scann"])
        )
