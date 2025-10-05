import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def generate_graphs():

    features_df = pd.read_csv('output/cleaned_flight_features_v2.csv')
    scores_df = pd.read_csv('output/test_ridhayneerajnathoo.csv')

    sns.set_style("whitegrid")

    # Top 10 Difficult Destinations
    plt.figure(figsize=(12, 8))
    difficult_flights = scores_df[scores_df['DifficultyClass'] == 'Difficult']
    destination_counts = difficult_flights['scheduled_arrival_station_code'].value_counts().head(10)
    sns.barplot(x=destination_counts.values, y=destination_counts.index, palette='viridis', orient='h')
    plt.title('Top 10 Destinations by Number of "Difficult" Flights', fontsize=16)
    plt.xlabel('Count of "Difficult" Flights')
    plt.ylabel('Destination Airport')
    plt.tight_layout()
    plt.savefig('output/2_top_difficult_destinations.png')

    # Driver Analysis for IAH
    target_destination = 'IAH'
    df_target = scores_df[(scores_df['scheduled_arrival_station_code'] == target_destination) & (scores_df['DifficultyClass'].isin(['Easy', 'Difficult']))]
    
    # Normalize data for better comparison on the same scale
    df_target['Normalized SSR'] = df_target['ssr_count'] / features_df['ssr_count'].max() * 100
    df_target['Normalized Ground Time'] = df_target['ground_time_cushion'] / features_df['ground_time_cushion'].max() * 100
    
    df_plot = df_target.groupby('DifficultyClass')[['Normalized SSR', 'Normalized Ground Time']].mean().reset_index()
    df_plot_melted = df_plot.melt(id_vars='DifficultyClass', var_name='Feature', value_name='Average Value')

    plt.figure(figsize=(10, 7))
    sns.barplot(data=df_plot_melted, x='Feature', y='Average Value', hue='DifficultyClass', palette={'Easy': 'skyblue', 'Difficult': 'tomato'})
    plt.title(f'Key Driver Comparison for Destination: {target_destination}', fontsize=16)
    plt.ylabel('Normalized Average Value (Higher is harder)')
    plt.xlabel('Difficulty Driver')
    plt.xticks(ticks=[0, 1], labels=['Special Service Requests (SSR)', 'Ground Time Pressure'])
    plt.tight_layout()
    plt.savefig('output/3_driver_analysis_iah.png')

if __name__ == '__main__':
    generate_graphs()