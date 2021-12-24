"""Functions in file from:
https://developers.google.com/machine-learning/recommendation/labs/movie-rec-programming-exercise

Original file from here: https://github.com/MarshallKrakauer/BGGAnalysis/blob/main/Collaborative_Filtering.ipynb
"""

from __future__ import print_function

import collections

import numpy as np
import pandas as pd
import tensorflow.compat.v1 as tf
from matplotlib import pyplot as plt
tf.compat.v1.disable_eager_execution()
tf.logging.set_verbosity(tf.logging.ERROR)

# Constants
DOT = 'dot'
COSINE = 'cosine'


def mask(df, key, function):
    """Returns a filtered dataframe, by applying function to key"""
    return df[function(df[key])]


def flatten_cols(df):
    df.columns = [' '.join(col).strip() for col in df.columns]
    return df


def split_dataframe(df, holdout_fraction=0.1):
    """Splits a DataFrame into training and test sets.
    Args:
      df: a dataframe.
      holdout_fraction: fraction of dataframe rows to use in the test set.
    Returns:
      train: dataframe for training
      test: dataframe for testing
    """
    test = df.sample(frac=holdout_fraction, replace=False)
    train = df[~df.index.isin(test.index)]
    return train, test


def build_rating_sparse_tensor(ratings_df):
    """
    Args:
      ratings_df: a pd.DataFrame with `user_id`, `movie_id` and `rating` columns.
    Returns:
      a tf.SparseTensor representing the ratings matrix.
    """
    indices = ratings_df[['user_idx', 'game_idx']].values
    values_ = ratings_df['rating_int'].values
    tensor_shape = [ratings_df['user_idx'].nunique(), ratings_df['game_idx'].nunique()]
    return tf.SparseTensor(
        indices=indices,
        values=values_,
        dense_shape=tensor_shape)


def sparse_mean_square_error(sparse_ratings, user_embeddings, movie_embeddings):
    """
    Args:
      sparse_ratings: A SparseTensor rating matrix, of dense_shape [N, M]
      user_embeddings: A dense Tensor U of shape [N, k] where k is the embedding
        dimension, such that U_i is the embedding of user i.
      movie_embeddings: A dense Tensor V of shape [M, k] where k is the embedding
        dimension, such that V_j is the embedding of movie j.
    Returns:
      A scalar Tensor representing the MSE between the true ratings and the
        model's predictions.
    """
    # print('SPARSE RATING INDICES:', sparse_ratings.indices)
    predictions = tf.gather_nd(
        tf.matmul(user_embeddings, movie_embeddings, transpose_b=True),
        sparse_ratings.indices)
    # [128812, 272]
    print('finished gather call')
    loss = tf.losses.mean_squared_error(sparse_ratings.values, predictions)
    return loss


# noinspection PyUnresolvedReferences
def build_model(ratings_dataframe, embedding_dim=3, init_stddev=1.):
    """
    Args:
      ratings_dataframe: a DataFrame of the ratings
      embedding_dim: the dimension of the embedding vectors.
      init_stddev: float, the standard deviation of the random initial embeddings.
    Returns:
      model: a CFModel.
    """
    # Split the ratings DataFrame into train and test.
    train_ratings_0, test_ratings_0 = split_dataframe(ratings_dataframe)
    # SparseTensor representation of the train and test datasets.
    a_train = build_rating_sparse_tensor(train_ratings_0)
    a_test = build_rating_sparse_tensor(test_ratings_0)
    # Initialize the embeddings using a normal distribution.
    U = tf.Variable(tf.random_normal(
        [a_train.dense_shape[0], embedding_dim], stddev=init_stddev))
    V = tf.Variable(tf.random_normal(
        [a_train.dense_shape[1], embedding_dim], stddev=init_stddev))
    train_loss = sparse_mean_square_error(a_train, U, V)
    test_loss = sparse_mean_square_error(a_test, U, V)
    metrics = {
        'train_error': train_loss,
        'test_error': test_loss
    }
    embeddings = {
        "user_idx": U,
        "game_idx": V
    }
    return CFModel(embeddings, train_loss, [metrics])


def game_neighbors(model_0, title_substring, games, measure=DOT, k=6):
    # Search for movie ids that match the given substring.
    ids = games.loc[games['title'].str.contains(title_substring), ['game_idx']].values
    ids = list(ids.flatten())
    titles = games.loc[games['game_idx'].isin(ids), 'title'].values
    if len(titles) == 0:
        raise ValueError("Found no games with title %s" % title_substring)
    print("Nearest neighbors of : %s." % titles[0])
    if len(titles) > 1:
        print("[Found more than one matching game. Other candidates: {}]".format(
            ", ".join(titles[1:])))
    movie_id = ids[0]
    scores = compute_scores(
        model_0.embeddings["game_idx"][movie_id], model_0.embeddings["game_idx"],
        measure)
    score_key = measure + ' score'
    df = pd.DataFrame({
        score_key: list(scores),
        'titles': games['title']
    })
    df.sort_values('score_key', ascending=False, inplace=True)
    print(df.head(k))


def compute_scores(query_embedding, item_embeddings, measure=DOT):
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
    u = query_embedding
    V = item_embeddings
    if measure == COSINE:
        V = V / np.linalg.norm(V, axis=1, keepdims=True)
        u = u / np.linalg.norm(u)
    scores = u.dot(V.T)
    return scores


class CFModel(object):
    """Simple class that represents a collaborative filtering model"""

    def __init__(self, embedding_vars, loss, metrics=None):
        """Initializes a CFModel.
        Args:
          embedding_vars: A dictionary of tf.Variables.
          loss: A float Tensor. The loss to optimize.
          metrics: optional list of dictionaries of Tensors. The metrics in each
            dictionary will be plotted in a separate figure during training.
        """
        self._embedding_vars = embedding_vars
        self._loss = loss
        self._metrics = metrics
        self._embeddings = {k: None for k in embedding_vars}
        self._session = None

    @property
    def embeddings(self):
        """The embeddings dictionary."""
        return self._embeddings

    def train(self, num_iterations=100, learning_rate=1.0, plot_results=True,
              optimizer=tf.train.GradientDescentOptimizer):
        """Trains the model.
        Args:
          num_iterations: number of iterations to run.
          learning_rate: optimizer learning rate.
          plot_results: whether to plot the results at the end of training.
          optimizer: the optimizer to use. Default to GradientDescentOptimizer.
        Returns:
          The metrics dictionary evaluated at the last iteration.
        """
        with self._loss.graph.as_default():
            opt = optimizer(learning_rate)
            train_op = opt.minimize(self._loss)
            local_init_op = tf.group(
                tf.variables_initializer(opt.variables()),
                tf.local_variables_initializer())
            if self._session is None:
                self._session = tf.Session()
                with self._session.as_default():
                    self._session.run(tf.global_variables_initializer())
                    self._session.run(tf.tables_initializer())
                    tf.train.start_queue_runners()

        with self._session.as_default():
            local_init_op.run()
            iterations = []
            metrics = self._metrics or ({},)
            metrics_vals = [collections.defaultdict(list) for _ in self._metrics]

            # Train and append results.
            for i in range(num_iterations + 1):
                _, results = self._session.run((train_op, metrics))
                if (i % 10 == 0) or i == num_iterations:
                    print("\r iteration %d: " % i + ", ".join(
                        ["%s=%f" % (k, v) for r in results for k, v in r.items()]),
                          end='')
                    iterations.append(i)
                    for metric_val, result in zip(metrics_vals, results):
                        for k, v in result.items():
                            metric_val[k].append(v)

            for k, v in self._embedding_vars.items():
                self._embeddings[k] = v.eval()

            if plot_results:
                # Plot the metrics.
                num_subplots = len(metrics) + 1
                fig = plt.figure()
                fig.set_size_inches(num_subplots * 10, 8)
                for i, metric_vals in enumerate(metrics_vals):
                    ax = fig.add_subplot(1, num_subplots, i + 1)
                    for k, v in metric_vals.items():
                        ax.plot(iterations, v, label=k)
                    ax.set_xlim([1, num_iterations])
                    ax.legend()
            return results


if __name__ == '__main__':
    ratings = pd.read_csv('ratings_data_0.csv').append(pd.read_csv('ratings_data_1.csv'))
    train_ratings, test_ratings = split_dataframe(ratings)
    # SparseTensor representation of the train and test datasets.
    A_train = build_rating_sparse_tensor(train_ratings)
    A_test = build_rating_sparse_tensor(test_ratings)
    # Build the CF model and train it.
    model = build_model(ratings, embedding_dim=10, init_stddev=0.5)
    model.train(num_iterations=200, learning_rate=40., plot_results=True)
