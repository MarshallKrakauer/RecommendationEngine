"""Ranking Model using TF Version 2.

Based on the code from this page: https://www.tensorflow.org/recommenders/examples/basic_ranking

Variable names currently refer to movies. This will be changed in future versions.
"""

import logging

import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_recommenders as tfrs


class RankingModel(tf.keras.Model):

    def get_config(self):
        pass

    def __init__(self):
        super().__init__()
        embedding_dimension = 32

        # Compute embeddings for users.
        self.user_embeddings = tf.keras.Sequential([
            tf.keras.layers.StringLookup(
                vocabulary=unique_user_ids, mask_token=None),
            tf.keras.layers.Embedding(len(unique_user_ids) + 1, embedding_dimension)
        ])

        # Compute embeddings for movies.
        self.movie_embeddings = tf.keras.Sequential([
            tf.keras.layers.StringLookup(
                vocabulary=unique_movie_titles, mask_token=None),
            tf.keras.layers.Embedding(len(unique_movie_titles) + 1, embedding_dimension)
        ])

        # Compute predictions.
        self.ratings = tf.keras.Sequential([
            # Learn multiple dense layers.
            tf.keras.layers.Dense(256, activation="relu"),
            tf.keras.layers.Dense(64, activation="relu"),
            # Make rating predictions in the final layer.
            tf.keras.layers.Dense(1)
        ])

    def __call__(self, inputs, **kwargs):
        user_id, movie_title = inputs

        user_embedding = self.user_embeddings(user_id)
        movie_embedding = self.movie_embeddings(movie_title)

        return self.ratings(tf.concat([user_embedding, movie_embedding], axis=1))


class BGGModel(tfrs.models.Model):

    def __init__(self):
        super().__init__()
        self.ranking_model: tf.keras.Model = RankingModel()
        self.task: tf.keras.layers.Layer = tfrs.tasks.Ranking(
            loss=tf.keras.losses.MeanSquaredError(),
            metrics=[tf.keras.metrics.RootMeanSquaredError()])

    def __call__(self, features):
        return self.ranking_model(
            (features["user_idx"], features["game_idx"]))

    def compute_loss(self, features, training=False):
        labels = features.pop("rating")

        rating_predictions = self(features)

        # The task computes the loss and the metrics.
        return self.task(labels=labels, predictions=rating_predictions)


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

    model = BGGModel()
    model.compile(optimizer=tf.keras.optimizers.Adagrad(learning_rate=0.1))
    cached_train = train.shuffle(100_000).batch(8192).cache()
    cached_test = test.batch(4096).cache()
    model.fit(cached_train, epochs=3)
    model.evaluate(cached_test, return_dict=True)

    test_ratings = {}
    test_movie_titles = ['244', "9", "273", "1", "278", "16"]
    for movie_title in test_movie_titles:
        test_ratings[movie_title] = model({
            "user_idx": np.array(["22"]),
            "game_idx": np.array([movie_title])
        })

    print("Ratings:")
    for title, score in sorted(test_ratings.items(), key=lambda x: x[1], reverse=True):
        print(f"{title}: {score}")