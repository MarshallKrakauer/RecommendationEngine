# Recommendation Engine File Guide
I will use this space to guide viewers through the files. This repository contains multiple recommendation enginges, all using the same underlying data. Each model takes in Board Game Geek (BGG) user's past ratings for games and recommends them a new set of games. 

## Setup
**CreateGameData.py** creates the CSV files seen in the **DataFiles** directory. These CSV files contain a reduced dataset scraped from the BGG API. For this, I levered the data from my BGG Analysis page

## Models

### Tensorflow V1
**TF1_CollaborativeFiltering.py** and **TF1_CollaborativeFiltering_Regularization.py** recommend games using TensorFlow version 1. The latter contains reglulraizion to prevent the model from only recommending the most popular games 

### Tenorflow V2
The remaining files emplyt the (as of right now), most up-to-date version TensorFlow: version 2. 

**RetrievalModel.py** takes a large swatch of game rating data, and reduces it to a small number of recommended items for a user. The **RankingModel.py** files takes a relatively small number of inputs and ranks them in terms of which ones it would most recommend to a user. The **MultiTaskModel.py** maximized both of these tasks in a single run.

Finally, the **ModelReader.py** imports one of the three V2 models, and outputs the results in a readable format. 
