"""File introduces regularization parameters to the collaborative filtering model."""

from __future__ import print_function

import pandas as pd
import numpy as np
import tensorflow.compat.v1 as tf

from TF1_CollaborativeFiltering import CFModel, sparse_mean_square_error
from TF1_CollaborativeFiltering import split_dataframe, build_rating_sparse_tensor

tf.compat.v1.disable_eager_execution()
tf.logging.set_verbosity(tf.logging.ERROR)


def display_top_neighbors(model_for_neighbors, title_substring, measure='dot', k=10):
    # Search for movie ids that match the given substring.
    ids = games.loc[games['title'].str.contains(title_substring), ['game_idx']].values
    ids = list(ids.flatten())
    titles = games.loc[games['game_idx'].isin(ids), 'title'].values
    if len(titles) == 0:
        raise ValueError("Found no games with title %s" % title_substring)
    print("\nNearest neighbors of : %s." % titles[0])
    if len(titles) > 1:
        print("[Found more than one matching game. Other candidates: {}]".format(
            ", ".join(titles[1:])))
    movie_id = ids[0]
    scores = compute_distance_scores(
        model_for_neighbors.embeddings["game_idx"][movie_id], model_for_neighbors.embeddings["game_idx"],
        measure)
    score_key = measure + ' score'
    df = pd.DataFrame({
        score_key: list(scores),
        'titles': games['title']
    })
    print(df.sort_values([score_key], ascending=False).head(k))


def gravity(U, V):
    """Creates a gravity loss given two embedding matrices."""
    return 1. / (U.shape[0] * V.shape[0]) * tf.reduce_sum(
        tf.matmul(U, U, transpose_a=True) * tf.matmul(V, V, transpose_a=True))


def compute_distance_scores(query_embedding, item_embeddings, measure='dot'):
    """Computes the scores of the candidates given a query.
    Args:
      query_embedding: a vector of shape [k], representing the query embedding.
      item_embeddings: a matrix of shape [N, k], such that row i is the embedding
        of item i.
      measure: a string specifying the similarity measure to be used. Can be
        either DOT or COSINE.
    Returns:
      scores: a vector of shape [N], such that scores[i] is the score of item i.
    """
    if measure == 'cosine':
        item_embeddings = item_embeddings / np.linalg.norm(item_embeddings, axis=1, keepdims=True)
        query_embedding = query_embedding / np.linalg.norm(query_embedding)
    scores = query_embedding.dot(item_embeddings.T)
    return scores

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
    display_top_neighbors(reg_model, "caverna", 'cosine', k=10)

