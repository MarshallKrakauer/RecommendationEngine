"""Retrieval Model using TF Version 2.

Based on the code from this page: https://www.tensorflow.org/recommenders/examples/basic_retrieval

Some code will contain "movie" terminology to match Google's code. This will be changed later.
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
SAVE_MODEL = True


class GameModel(tfrs.Model):

    def __init__(self):
        super().__init__()

        self.title_embedding = tf.keras.Sequential([
            tf.keras.layers.StringLookup(
                vocabulary=unique_game_titles),
            tf.keras.layers.Embedding(len(unique_game_titles) + 1, DIMENSIONS)])

        # Make year variable discrete
        self.discrete_years = tf.keras.layers.Discretization(num_bins=YEAR_BINS)
        self.year_embedding = tf.keras.Sequential([
            self.discrete_years,
            tf.keras.layers.Embedding(YEAR_BINS, 10, mask_zero=True)])
        self.discrete_years.adapt(game_years)

        # Make Num of Ratings variable discrete
        self.discrete_num_ratings = tf.keras.layers.Discretization(num_bins=YEAR_BINS)
        self.num_ratings_embedding = tf.keras.Sequential([
            self.discrete_num_ratings,
            tf.keras.layers.Embedding(YEAR_BINS, 10, mask_zero=True)])
        self.discrete_num_ratings.adapt(game_num_ratings)

    def call(self, inputs):
        return tf.concat([
            self.title_embedding(inputs['title']),
            self.year_embedding(inputs['year']),
            self.num_ratings_embedding(inputs['num_ratings']),
        ], axis=1)


class UserModel(tf.keras.Model):

    def __init__(self):
        super().__init__()

        self.user_embedding = tf.keras.Sequential([
            tf.keras.layers.StringLookup(
                vocabulary=unique_user_ids, mask_token=None),
            tf.keras.layers.Embedding(len(unique_user_ids) + 1, 10), ])

        self.year_embedding = tf.keras.Sequential([
            tf.keras.layers.Discretization(num_bins=YEAR_BINS),
            tf.keras.layers.Embedding(YEAR_BINS, 10),
        ])

        self.num_rating_embedding = tf.keras.Sequential([
            tf.keras.layers.Discretization(num_bins=YEAR_BINS),
            tf.keras.layers.Embedding(YEAR_BINS, 10),
        ])

    def call(self, inputs):
        return tf.concat([
            self.user_embedding(inputs["user_idx"]),
            self.year_embedding(inputs["year"]),
            self.num_rating_embedding(inputs["num_ratings"]),
        ], axis=1)


class BGGRetrievalModel(tfrs.Model):

    def __init__(self, layer_sizes):
        super().__init__()
        self.query_model = QueryModel(layer_sizes)
        self.candidate_model = CandidateModel(layer_sizes)
        self.task = tfrs.tasks.Retrieval(
            metrics=tfrs.metrics.FactorizedTopK(
                candidates=games.batch(64).map(self.candidate_model), ), )

    def compute_loss(self, features: Dict[Text, tf.Tensor], training=False) -> tf.Tensor:
        # We pick out the user features and pass them into the user model.
        user_embeddings = self.query_model({'user_idx': features['user_idx'],
                                            'year': features['year'],
                                            'num_ratings': features['num_ratings']})

        # And pick out the movie features and pass them into the movie model,
        # getting embeddings back.
        positive_movie_embeddings = self.candidate_model({'title': features['title'],
                                                          'year': features['year'],
                                                          'num_ratings': features['num_ratings']})

        # The task computes the loss and the metrics.
        return self.task(user_embeddings, positive_movie_embeddings, compute_metrics=not training)


class CandidateModel(tf.keras.Model):
    """Model for encoding movies."""

    def __init__(self, layer_sizes):
        """Model for encoding movies.

        Args:
          layer_sizes:
            A list of integers where the i-th entry represents the number of units
            the i-th layer contains.
        """
        super().__init__()

        self.embedding_model = GameModel()

        # Then construct the layers.
        self.dense_layers = tf.keras.Sequential()

        # Use the ReLU activation for all but the last layer.
        for layer_size in layer_sizes[:-1]:
            self.dense_layers.add(tf.keras.layers.Dense(layer_size, activation="relu"))

        # No activation for the last layer.
        for layer_size in layer_sizes[-1:]:
            self.dense_layers.add(tf.keras.layers.Dense(layer_size))

    def call(self, inputs):
        feature_embedding = self.embedding_model(inputs)
        return self.dense_layers(feature_embedding)


class QueryModel(tf.keras.Model):
    """Model for encoding user queries."""

    def __init__(self, layer_sizes):
        """Model for encoding user queries.

        Args:
          layer_sizes:
            A list of integers where the i-th entry represents the number of units
            the i-th layer contains.
        """
        super().__init__()

        # We first use the user model for generating embeddings.
        self.embedding_model = UserModel()

        # Then construct the layers.
        self.dense_layers = tf.keras.Sequential()

        # Use the ReLU activation for all but the last layer.
        for layer_size in layer_sizes[:-1]:
            self.dense_layers.add(tf.keras.layers.Dense(layer_size, activation="relu"))

        # No activation for the last layer.
        for layer_size in layer_sizes[-1:]:
            self.dense_layers.add(tf.keras.layers.Dense(layer_size))

    def call(self, inputs):
        feature_embedding = self.embedding_model(inputs)
        return self.dense_layers(feature_embedding)


def get_ratings_data():
    """
    Imports ratings dataframe with user index, game index, rating, and title

    :return: tf.Dataset
        Dataset of ratings with user and game label
    """
    ratings_original = pd.read_csv('DataFiles/ratings_data_0.csv').append(pd.read_csv('DataFiles/ratings_data_1.csv'))
    ratings_original = ratings_original[['rating_int', 'user_idx', 'game_idx']]
    title_join = pd.read_csv('DataFiles/games.csv')
    title_join = title_join[['game_idx', 'title', 'year', 'ratings']]
    title_join.rename(columns={'ratings': 'num_ratings'}, inplace=True)
    title_join.drop_duplicates(inplace=True)
    ratings_original = ratings_original.merge(title_join, on='game_idx')
    ratings_original.rename(columns={'rating_int': 'rating'}, inplace=True)
    value_dict = {name: values for name, values in ratings_original.items()}
    value_dict['rating'] = np.asarray(value_dict['rating'].astype('float32'))
    value_dict['user_idx'] = np.asarray(value_dict['user_idx'].astype('string'))
    value_dict['game_idx'] = np.asarray(value_dict['game_idx'].astype('string'))
    value_dict['year'] = np.asarray(value_dict['year'].astype('float32'))
    value_dict['num_ratings'] = np.asarray(value_dict['num_ratings'].astype('float32'))
    ratings_tf = tf.data.Dataset.from_tensor_slices(value_dict)
    return ratings_tf.shuffle(4_000_000, seed=0, reshuffle_each_iteration=False)


def get_games_data():
    """
    Create tensorflow data for information on individual board games

    :return: iterable
        List of game titles
    """
    games_raw = pd.read_csv('DataFiles/games.csv')
    games_raw.rename(columns={'rating': 'bgg_rating', 'ratings': 'num_ratings'}, inplace=True)
    games_raw = games_raw[['title', 'year', 'bgg_rating', 'num_ratings']]

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
    max_, min_ = games_raw['num_ratings'].max(), games_raw['num_ratings'].min()
    games_raw['num_ratings'] = games_raw['num_ratings'] - min_
    games_raw['num_ratings'] = games_raw['num_ratings'] / (max_ - min_)

    games_raw['title'] = games_raw['title'].astype('string')
    games_raw['year'] = games_raw['year'].astype('float32')
    games_raw['bgg_rating'] = games_raw['bgg_rating'].astype('float32')
    games_raw['num_ratings'] = games_raw['num_ratings'].astype('float32')

    value_dict = {name: values for name, values in games_raw.items()}
    value_dict['title'] = np.asarray(value_dict['title'].astype('string'))
    value_dict['year'] = np.asarray(value_dict['year'].astype('float32'))
    value_dict['bgg_rating'] = np.asarray(value_dict['bgg_rating'].astype('float32'))
    value_dict['num_ratings'] = np.asarray(value_dict['num_ratings'].astype('float32'))

    final_data = tf.data.Dataset.from_tensor_slices(value_dict)

    return final_data


if __name__ == '__main__':
    tf.get_logger().setLevel(logging.ERROR)
    tf.random.set_seed(0)

    # Obtain games info
    game_info_original = get_games_data()
    games = game_info_original

    game_titles = game_info_original.map(lambda x: x['title'])
    game_years = game_info_original.map(lambda x: x['year'])
    game_num_ratings = game_info_original.map(lambda x: x['num_ratings'])

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

    retrieval_model = BGGRetrievalModel([128, 64])

    # Fit Model
    retrieval_model.compile(optimizer=tf.keras.optimizers.Adagrad(learning_rate=0.5))
    cached_train = ratings_train.shuffle(1_000_000, seed=0).batch(5_000).cache()
    cached_test = ratings_test.batch(10_000).cache()
    retrieval_model.fit(cached_train, epochs=1)

    # Evaluate Model
    retrieval_model.evaluate(cached_test, return_dict=True)

    if SAVE_MODEL:
        brute_force = tfrs.layers.factorized_top_k.BruteForce(retrieval_model.query_model)

        brute_force.index_from_dataset(
            tf.data.Dataset.zip((game_titles.batch(20), games.batch(20).map(retrieval_model.candidate_model))))

        test = brute_force({"user_idx": np.array(["32"]), "year": np.array([0]), 'num_ratings': np.array([0])})

        print(test)

        tf.saved_model.save(brute_force, PATH, options=tf.saved_model.SaveOptions())

        # loaded = tf.saved_model.load(PATH)
