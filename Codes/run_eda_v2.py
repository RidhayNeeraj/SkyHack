import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def perform_final_eda():

    master_df = pd.read_csv('output/cleaned_flight_features_v2.csv')
    
    results_summary = []
    results_summary.append("Final Exploratory Data Analysis (EDA) Results")
    results_summary.append("="*50)

    # Question 1: Delay Analysis
    avg_delay = master_df['departure_delay'].mean()
    late_percentage = (master_df['departure_delay'] > 0).mean() * 100
    results_summary.append("\n1. Departure Delay Analysis:")
    results_summary.append(f"   Average departure delay: {avg_delay:.2f} minutes")
    results_summary.append(f"   Percentage of flights departing late: {late_percentage:.2f}%")

    # Question 2: Ground Time Analysis
    below_minimum_flights = (master_df['ground_time_cushion'] <= 0).sum()
    total_flights = len(master_df)
    below_min_percentage = (below_minimum_flights / total_flights) * 100
    results_summary.append("\n2. Ground Time Analysis:")
    results_summary.append(f"   Flights with ground time at or below minimum: {below_minimum_flights} flights")
    results_summary.append(f"   Percentage of total flights: {below_min_percentage:.2f}%")

    # Question 3: Baggage Analysis (Now including Hot Transfers)
    avg_transfer_ratio = master_df['transfer_ratio'].mean()
    avg_hot_transfer_ratio = master_df['hot_transfer_ratio'].mean()
    results_summary.append("\n3. Baggage Analysis:")
    results_summary.append(f"   Average ratio of regular transfer bags to total bags: {avg_transfer_ratio:.2f}")
    results_summary.append(f"   Average ratio of HOT transfer bags to total bags: {avg_hot_transfer_ratio:.2f}")

    # Question 4: Passenger Load vs. Difficulty Analysis
    correlation = master_df['load_factor'].corr(master_df['departure_delay'])
    results_summary.append("\n4. Passenger Load vs. Difficulty Analysis:")
    results_summary.append(f"   Correlation between load factor and departure delay: {correlation:.3f}")

    # Question 5: SSR vs. Delay Analysis
    master_df['load_category'] = pd.qcut(master_df['load_factor'], 2, labels=['Low Load', 'High Load'])
    master_df['ssr_category'] = np.where(master_df['ssr_count'] > master_df['ssr_count'].median(), 'High SSR', 'Low SSR')
    delay_by_group = master_df.groupby(['load_category', 'ssr_category'])['departure_delay'].mean().unstack()
    results_summary.append("\n5. Special Service Requests (SSR) vs. Delay Analysis:")
    results_summary.append("   - Average Delay (in minutes) by Load and SSR count:")
    results_summary.append(delay_by_group.round(2).to_string())

    # Saving the results to a text file
    with open('output/eda_results_v2.txt', 'w') as f:
        for line in results_summary:
            f.write(line + '\n')

    # Generating and saving final plots
    sns.set_style("whitegrid")
    
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=master_df, x='load_factor', y='departure_delay', alpha=0.4)
    plt.title('Final: Passenger Load Factor vs. Departure Delay', fontsize=16)
    plt.savefig('output/final_v2_load_factor_vs_delay.png')
    
    delay_by_group.plot(kind='bar', figsize=(10, 7), rot=0)
    plt.title('Final: Average Delay by Load and SSR Count', fontsize=16)
    plt.savefig('output/final_v2_ssr_and_load_vs_delay.png')

if __name__ == '__main__':
    perform_final_eda()