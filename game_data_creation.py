"""Data created from this repository and written to csv:
 https://github.com/MarshallKrakauer/BGGAnalysis """

import pandas as pd
import numpy as np

if __name__ == '__main__':
    game_data = pd.read_csv('game_info.csv')
    print(game_data.iloc[0, :])
    category_li = list(game_data['category'].unique())
    mechanic_li = list(game_data['mechanic'].unique())
    print(category_li)
