"""Ranking Model using TF Version 2.

Based on the code from this page: https://www.tensorflow.org/recommenders/examples/basic_ranking
"""

import numpy as np
import pandas as pd
import tensorflow as tf
import copy


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

    def call(self, inputs, **kwargs):
        user_id, movie_title = inputs

        user_embedding = self.user_embeddings(user_id)
        movie_embedding = self.movie_embeddings(movie_title)

        return self.ratings(tf.concat([user_embedding, movie_embedding], axis=1))


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    tf.autograph.set_verbosity(0)
    tf.random.set_seed(0)

    ratings = pd.read_csv('DataFiles/ratings_data_0.csv').append(pd.read_csv('DataFiles/ratings_data_1.csv'))
    ratings = ratings[['ratings', 'user_idx', 'game_idx']]
    original_ratings = copy.deepcopy(ratings)
    value_dict = {name: values for name, values in ratings.items()}
    value_dict['ratings'] = np.asarray(value_dict['ratings'].astype('float32'))
    value_dict['user_idx'] = np.asarray(value_dict['user_idx'].astype('string'))
    value_dict['game_idx'] = np.asarray(value_dict['game_idx'].astype('string'))
    ratings = tf.data.Dataset.from_tensor_slices(value_dict)
    shuffled = ratings.shuffle(400_000, seed=0, reshuffle_each_iteration=False)
    train = shuffled.take(80_000)
    test = shuffled.skip(80_000).take(20_000)
    movie_titles = ratings.batch(1_000_000).map(lambda x: x["game_idx"])
    user_ids = ratings.batch(1_000_000).map(lambda x: x["user_idx"])

    unique_movie_titles = np.unique(np.concatenate(list(movie_titles)))
    unique_user_ids = np.unique(np.concatenate(list(user_ids)))
    print(RankingModel()((['0'], ['0'])))

