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
DIMENSIONS = 12
SAVE_MODEL = False


class BGGRetrievalModel(tfrs.Model):

    def __init__(self, user_model_retrieval, movie_model_retrieval, task):
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


def get_ratings_data():
    """
    Imports ratings dataframe with user index, game index, rating, and title

    :return: tf.Dataset
        Dataset of ratings with user and game label
    """

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
    """
    Create tensorflow data for information on individual board games

    :return: iterable
        List of game titles
    """
    games_raw = pd.read_csv('DataFiles/games.csv')
    games_raw = games_raw[['title', 'year', 'rating']]
    games_raw['title'] = games_raw['title'].astype('string')
    games_raw['year'] = games_raw['year'].astype('int32')
    games_raw['rating'] = games_raw['rating'].astype('int32')
    # value_dict = {name: values for name, values in games_small.items()}  # Not currently in use
    game_idx_vals = tf.data.Dataset.from_tensor_slices(list(games_raw['title'].values))

    '''
    Commented out. Side features may be used in more advanced model
    year_vals = tf.data.Dataset.from_tensor_slices(list(games_raw['year'].values))
    rating_vals = tf.data.Dataset.from_tensor_slices(list(games_raw['rating'].values))
    data = tf.data.Dataset.zip((game_idx_vals, year_vals, rating_vals))
    '''

    return game_idx_vals


def get_setup_info():
    """
    Fetches train and test data for the retrieval model, as well as the internal user and game model


    :return:
        ratings_train tf Dataset with rating data for training
        ratings_test tf Dataset with rating data for testing
        bgg_user_model Sequential model to embed user info
        bgg_game_model Sequential model to embed game info
    """

    ratings = get_ratings_data()
    ratings_train = ratings.take(3_000_000)
    ratings_test = ratings.skip(3_000_000).take(800_000)
    movie_titles = ratings.batch(300).map(lambda x: x["title"])
    user_ids = ratings.batch(1_000_000).map(lambda x: x["user_idx"])

    unique_movie_titles = np.unique(np.concatenate(list(movie_titles)))
    unique_user_ids = np.unique(np.concatenate(list(user_ids)))

    embedding_dimension = DIMENSIONS

    bgg_user_model = tf.keras.Sequential([
        tf.keras.layers.StringLookup(
            vocabulary=unique_user_ids, mask_token=None),
        # We add an additional embedding to account for unknown tokens.
        tf.keras.layers.Embedding(len(unique_user_ids) + 1, embedding_dimension)])

    bgg_game_model = tf.keras.Sequential([
        tf.keras.layers.StringLookup(
            vocabulary=unique_movie_titles, mask_token=None),
        tf.keras.layers.Embedding(len(unique_movie_titles) + 1, embedding_dimension)])

    return ratings_train, ratings_test, bgg_user_model, bgg_game_model


def train_tensorflow_model(train, test, user_model, movie_model, games):
    """
    Given input data, trains the retrieval model that can select the top recommended games for a user.

    :param train: tf Dataset
     dataset to model on
    :param test: tf Dataset
        dataset from which to judge model fit
    :param user_model: tf Sequential Model
        Lower dimensional embedding of user data
    :param movie_model: tf Sequential Model
        Lower dimensional embedding of game data
    :param games: pd Dataframe
        Data containing index, title, and other info for games
    :return: BGGRetrievalModel object
        Retrieval model that selects top N recommended games for a user
    """
    metrics = tfrs.metrics.FactorizedTopK(
        candidates=games.batch(64).map(movie_model))

    task = tfrs.tasks.Retrieval(metrics=metrics)

    retrieval_model = BGGRetrievalModel(user_model, movie_model, task)
    retrieval_model.compile(optimizer=tf.keras.optimizers.Adagrad(learning_rate=0.1))

    cached_train = train.shuffle(1_000_000).batch(5_000).cache()
    cached_test = test.batch(4096).cache()

    retrieval_model.fit(cached_train, epochs=EPOCHS)

    return_dict = retrieval_model.evaluate(cached_test, return_dict=True)
    print('~~~Evaluation Results~~~')
    print(return_dict)
    return retrieval_model


if __name__ == '__main__':
    tf.get_logger().setLevel(logging.ERROR)
    tf.random.set_seed(0)

    train_0, test_0, user_model_0, game_model_0 = get_setup_info()
    games_dataframe = get_games_data()
    model = train_tensorflow_model(train_0, test_0, user_model_0, game_model_0, games_dataframe)

    if SAVE_MODEL:
        # Create a model that takes in raw query features, and
        index = tfrs.layers.factorized_top_k.BruteForce(model.user_model)
        # recommends movies out of the entire movies dataset.
        index.index_from_dataset(
            tf.data.Dataset.zip((games.batch(100), games.batch(100).map(model.movie_model)))
        )

        # Save the index.
        tf.saved_model.save(index, PATH, options=tf.saved_model.SaveOptions())
