import pandas as pd

df_scores = pd.read_csv('output/test_ridhayneerajnathoo.csv')

TARGET_DESTINATION = 'IAH'

# Filter for just the flights to target destination
df_target = df_scores[df_scores['scheduled_arrival_station_code'] == TARGET_DESTINATION]

driver_analysis = df_target.groupby('DifficultyClass')[['ssr_count', 'ground_time_cushion', 'transfer_ratio', 'total_bags']].mean()

try:
    difficult_easy_comparison = driver_analysis.loc['Difficult'] - driver_analysis.loc['Easy']
    print("\nAverage Profile of 'Difficult' vs. 'Easy' Flights")
    print(driver_analysis.round(2))
    print("\nDifference ('Difficult' minus 'Easy')")
    print(difficult_easy_comparison.round(2))
except KeyError:
    print("\nCould not find both 'Difficult' and 'Easy' classes for this destination to make a comparison.")