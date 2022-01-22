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
EPOCHS = 1
DIMENSIONS = 10
YEAR_BINS = 20
GAMES_BATCH = 20
SAVE_MODEL = False


class GameModel(tfrs.Model):

    def __init__(self):
        super().__init__()

        self.title_embedding = tf.keras.Sequential([
            tf.keras.layers.StringLookup(
                vocabulary=unique_game_titles),
            tf.keras.layers.Embedding(len(unique_game_titles) + 1, DIMENSIONS)
        ])

        self.title_vectorizer = tf.keras.layers.Discretization(num_bins=YEAR_BINS)

        self.year_embedding = tf.keras.Sequential([
            self.title_vectorizer,
            tf.keras.layers.Embedding(YEAR_BINS, 10, mask_zero=True),
        ])

        self.title_vectorizer.adapt(game_years)

    def call(self, inputs):
        return tf.concat([
            self.title_embedding(inputs['title']),
            self.year_embedding(inputs['year']),
        ], axis=1)


class UserModel(tf.keras.Model):

    def __init__(self):
        super().__init__()

        self.user_embedding = tf.keras.Sequential([
            tf.keras.layers.StringLookup(
                vocabulary=unique_user_ids, mask_token=None),
            tf.keras.layers.Embedding(len(unique_user_ids) + 1, 10), ])

        self.year_embedding = tf.keras.Sequential([
            tf.keras.layers.Discretization(num_bins=15),
            tf.keras.layers.Embedding(15, 10),
        ])

    def call(self, inputs):
        return tf.concat([
            self.user_embedding(inputs["user_idx"]),
            self.year_embedding(inputs["year"]),
        ], axis=1)


class BGGRetrievalModel(tfrs.Model):

    def __init__(self, user_model_retrieval, movie_model_retrieval, task):
        super().__init__()
        self.movie_model: tf.keras.Model = movie_model_retrieval
        self.user_model: tf.keras.Model = user_model_retrieval
        self.task: tf.keras.layers.Layer = task

    def compute_loss(self, features: Dict[Text, tf.Tensor], training=False) -> tf.Tensor:
        # We pick out the user features and pass them into the user model.
        user_embeddings = self.user_model({'user_idx': features['user_idx'], 'year': features['year']})

        # And pick out the movie features and pass them into the movie model,
        # getting embeddings back.
        positive_movie_embeddings = self.movie_model({'title': features['title'], 'year': features['year']})

        # The task computes the loss and the metrics.
        return self.task(user_embeddings, positive_movie_embeddings)


def get_ratings_data():
    """
    Imports ratings dataframe with user index, game index, rating, and title

    :return: tf.Dataset
        Dataset of ratings with user and game label
    """
    ratings_original = pd.read_csv('DataFiles/ratings_data_0.csv').append(pd.read_csv('DataFiles/ratings_data_1.csv'))
    ratings_original = ratings_original[['rating_int', 'user_idx', 'game_idx']]
    title_join = pd.read_csv('DataFiles/games.csv')
    title_join = title_join[['game_idx', 'title', 'year']]
    title_join.drop_duplicates(inplace=True)
    ratings_original = ratings_original.merge(title_join, on='game_idx')
    ratings_original.rename(columns={'rating_int': 'rating'}, inplace=True)
    value_dict = {name: values for name, values in ratings_original.items()}
    value_dict['rating'] = np.asarray(value_dict['rating'].astype('float32'))
    value_dict['user_idx'] = np.asarray(value_dict['user_idx'].astype('string'))
    value_dict['game_idx'] = np.asarray(value_dict['game_idx'].astype('string'))
    value_dict['year'] = np.asarray(value_dict['year'].astype('float32'))
    ratings_tf = tf.data.Dataset.from_tensor_slices(value_dict)
    return ratings_tf.shuffle(4_000_000, seed=0, reshuffle_each_iteration=False)


def get_games_data():
    """
    Create tensorflow data for information on individual board games

    :return: iterable
        List of game titles
    """
    games_raw = pd.read_csv('DataFiles/games.csv')
    games_raw.rename(columns={'rating': 'bgg_rating'}, inplace=True)
    games_raw = games_raw[['title', 'year', 'bgg_rating', 'ratings']]

    # Fix years
    games_raw.loc[games_raw['year'] <= 1980, 'year'] = 1980
    games_raw.loc[games_raw['year'] > 2050, 'year'] = 1980  # Would only occur for very old games, would be BC

    # Min/Max Scale the year
    max_, min_ = games_raw['year'].max(), games_raw['year'].min()
    games_raw['year'] = games_raw['year'] - min_
    games_raw['year'] = games_raw['year'] / (max_ - min_)

    # Min/Max Scale the rating
    max_, min_ = games_raw['bgg_rating'].max(), games_raw['bgg_rating'].min()
    games_raw['bgg_rating'] = games_raw['bgg_rating'] - min_
    games_raw['bgg_rating'] = games_raw['bgg_rating'] / (max_ - min_)

    # Min/Max scale the ratings (num of ratings)
    max_, min_ = games_raw['ratings'].max(), games_raw['ratings'].min()
    games_raw['ratings'] = games_raw['ratings'] - min_
    games_raw['ratings'] = games_raw['ratings'] / (max_ - min_)

    games_raw['title'] = games_raw['title'].astype('string')
    games_raw['year'] = games_raw['year'].astype('float32')
    games_raw['bgg_rating'] = games_raw['bgg_rating'].astype('float32')
    games_raw['ratings'] = games_raw['ratings'].astype('float32')

    value_dict = {name: values for name, values in games_raw.items()}
    value_dict['title'] = np.asarray(value_dict['title'].astype('string'))
    value_dict['year'] = np.asarray(value_dict['year'].astype('float32'))
    value_dict['bgg_rating'] = np.asarray(value_dict['bgg_rating'].astype('float32'))
    value_dict['ratings'] = np.asarray(value_dict['ratings'].astype('float32'))

    # final_data = tf.data.Dataset.zip((titles, years, ratings, num_ratings))

    final_data = tf.data.Dataset.from_tensor_slices(value_dict)

    return final_data


if __name__ == '__main__':
    # Obtain games info
    game_info_original = get_games_data()
    games = game_info_original

    game_titles = game_info_original.map(lambda x: x['title'])
    game_years = game_info_original.map(lambda x: x['year'])



    #  Obtain user ratings and organize for tensorflow
    ratings = get_ratings_data()
    ratings_train = ratings.take(3_000_000)
    ratings_test = ratings.skip(3_000_000).take(800_000)
    movie_titles = ratings.batch(GAMES_BATCH).map(lambda x: x["title"])
    user_ids = ratings.batch(1_000).map(lambda x: x["user_idx"])

    unique_game_titles = np.unique(np.concatenate(list(movie_titles)))
    unique_user_ids = np.unique(np.concatenate(list(user_ids)))

    # Set up query and game model
    bgg_user_model = UserModel()
    bgg_game_model = GameModel()

    metrics = tfrs.metrics.FactorizedTopK(
        candidates=games.batch(64).map(bgg_game_model))

    task = tfrs.tasks.Retrieval(metrics=metrics)

    retrieval_model = BGGRetrievalModel(bgg_user_model, bgg_game_model, task)

    # Fit Model
    retrieval_model.compile(optimizer=tf.keras.optimizers.Adagrad(learning_rate=0.1))
    cached_train = ratings_train.shuffle(1_000_000, seed=0).batch(5_000).cache()
    cached_test = ratings_test.batch(10_000).cache()
    retrieval_model.fit(cached_train, epochs=1)

    # Evaluate Model
    retrieval_model.evaluate(cached_test, return_dict=True)

    if SAVE_MODEL:
        # Create a model that takes in raw query features, and
        index = tfrs.layers.factorized_top_k.BruteForce(model.user_model)
        # recommends movies out of the entire movies dataset.
        index.index_from_dataset(tf.data.Dataset.zip((games_tf_data.batch(GAMES_BATCH),
                                                      games_tf_data.batch(GAMES_BATCH).map(model.movie_model))))

        # Save the index.
        tf.saved_model.save(index, PATH, options=tf.saved_model.SaveOptions())
