import numpy as np
import pandas as pd

filepath: str = './data/movies.csv'

full_data = pd.read_csv(filepath)
# get rid of things we absolutely don't need
full_data.drop(labels=[
    'original_language',
    'production_companies',
    'production_countries',
    'homepage',
    'spoken_languages',
    'status',
    'tagline',
    'original_title',
    'crew',
], axis=1, inplace=True)
# drop null stuff
full_data.dropna(inplace=True)
full_data.to_csv('./data/filtered_movies.csv', index=False)