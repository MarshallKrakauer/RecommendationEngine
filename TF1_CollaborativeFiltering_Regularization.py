"""File will introduce regularization parameters to collaborative filtering model."""

from __future__ import print_function

import pandas as pd
import tensorflow.compat.v1 as tf

from TF1_CollaborativeFiltering import CFModel, sparse_mean_square_error
from TF1_CollaborativeFiltering import split_dataframe, build_rating_sparse_tensor

tf.compat.v1.disable_eager_execution()
tf.logging.set_verbosity(tf.logging.ERROR)


def gravity(U, V):
    """Creates a gravity loss given two embedding matrices."""
    return 1. / (U.shape[0] * V.shape[0]) * tf.reduce_sum(
        tf.matmul(U, U, transpose_a=True) * tf.matmul(V, V, transpose_a=True))


def build_regularized_model(
        ratings, embedding_dim=3, regularization_coeff=.1, gravity_coeff=1.,
        init_stddev=0.1):
    """
    Args:
      ratings: the DataFrame of movie ratings.
      embedding_dim: The dimension of the embedding space.
      regularization_coeff: The regularization coefficient lambda.
      gravity_coeff: The gravity regularization coefficient lambda_g.
      init_stddev: float, the standard deviation of the random initial embeddings.
    Returns:
      A CFModel object that uses a regularized loss.
    """
    # Split the ratings DataFrame into train and test.
    train_ratings, test_ratings = split_dataframe(ratings)
    # SparseTensor representation of the train and test datasets.
    A_train = build_rating_sparse_tensor(train_ratings)
    A_test = build_rating_sparse_tensor(test_ratings)
    U = tf.Variable(tf.random_normal(
        [A_train.dense_shape[0], embedding_dim], stddev=init_stddev))
    V = tf.Variable(tf.random_normal(
        [A_train.dense_shape[1], embedding_dim], stddev=init_stddev))

    error_train = sparse_mean_square_error(A_train, U, V)
    error_test = sparse_mean_square_error(A_test, U, V)
    gravity_loss = gravity_coeff * gravity(U, V)
    regularization_loss = regularization_coeff * (
            tf.reduce_sum(U * U) / U.shape[0] + tf.reduce_sum(V * V) / V.shape[0])
    total_loss = error_train + regularization_loss + gravity_loss
    losses = {
        'train_error_observed': error_train,
        'test_error_observed': error_test,
    }
    loss_components = {
        'observed_loss': error_train,
        'regularization_loss': regularization_loss,
        'gravity_loss': gravity_loss,
    }
    embeddings = {"user_idx": U, "game_idx": V}

    return CFModel(embeddings, total_loss, [losses, loss_components])


if __name__ == '__main__':
    ratings = pd.read_csv('DataFiles/ratings_data_0.csv').append(pd.read_csv('DataFiles/ratings_data_1.csv'))
    games = pd.read_csv('DataFiles/games.csv')
    train_ratings, test_ratings = split_dataframe(ratings)

    reg_model = build_regularized_model(
        ratings, regularization_coeff=0.1, gravity_coeff=1.0, embedding_dim=35,
        init_stddev=0.5)
    reg_model.train(num_iterations=100, learning_rate=40.)
