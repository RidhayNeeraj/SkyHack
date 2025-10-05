import pandas as pd

df_scores = pd.read_csv('output/test_ridhayneerajnathoo.csv')

difficult_flights = df_scores[df_scores['DifficultyClass'] == 'Difficult']
destination_counts = difficult_flights['scheduled_arrival_station_code'].value_counts()

print(destination_counts.head(5))