# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # import tensorflow as tf
    # import tensorflow_recommenders as tfrs
    import pandas as pd
    df = pd.read_csv('player_ratings.csv')
    print(df.head())

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
