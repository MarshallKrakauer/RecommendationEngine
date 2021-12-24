"""Data created from this repository and written to csv:
 https://github.com/MarshallKrakauer/BGGAnalysis """

import pandas as pd
import numpy as np

pd.options.mode.chained_assignment = None  # default='warn'


def create_attribute_columns(dataframe, attribute):
    new_dataframe = dataframe[['game_id', attribute]]
    col_list = []
    for attribute_name in iter(new_dataframe[attribute].unique()):
        column_name = clean_attribute(attribute_name)
        new_dataframe[column_name] = 0
        new_dataframe[column_name] = np.where(new_dataframe[attribute] == attribute_name, 1, 0)
        col_list.append(column_name)

    return pd.pivot_table(new_dataframe, index='game_id', values=col_list, aggfunc='max').reset_index()


def clean_attribute(name_string):
    try:
        name_string = name_string.replace(' ', '')
        name_string = name_string.replace(':', '')
        name_string = name_string.replace('/', '')
        name_string = name_string.replace(',', '')
        name_string = name_string.replace('-', '')
        name_string = name_string.replace('&', '')
    except AttributeError:
        return 'Other'

    return name_string


def create_game_data():
    game_data_initial = pd.read_csv('game_info.csv')
    game_constants = game_data_initial[['game_id', 'title', 'rating', 'bayes_rating', 'ratings', 'year']]
    mechanic_df = create_attribute_columns(game_data_initial, 'mechanic')
    category_df = create_attribute_columns(game_data_initial, 'category')
    game_constants = game_constants.merge(mechanic_df, on='game_id')
    game_constants = game_constants.merge(category_df, on='game_id')
    return game_constants
