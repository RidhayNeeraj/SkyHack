import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def visualize_eda():
    master_df = pd.read_csv('output/cleaned_flight_features_v2.csv')

    sns.set_style("whitegrid")

    plt.figure(figsize=(12, 7))
    sns.histplot(master_df['ground_time_cushion'], binwidth=5, kde=False)
    plt.axvline(x=0, color='red', linestyle='--', linewidth=2.5)
    plt.title('Zoomed-In View: Ground Time Cushion Distribution', fontsize=18)
    plt.xlabel('Ground Time Cushion (Minutes)', fontsize=12)
    plt.ylabel('Number of Flights', fontsize=12)
    plt.text(-45, 250, 'High-Risk Zone:\nFlights with Negative Cushion', color='red', ha='left', fontsize=12)
    plt.xlim(-50, 100)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('output/ground_time_distribution.png')

    avg_composition = {
        'Hot Transfer': master_df['hot_transfer_ratio'].mean(),
        'Regular Transfer': master_df['transfer_ratio'].mean(),
        'Origin': 1 - master_df['hot_transfer_ratio'].mean() - master_df['transfer_ratio'].mean()
    }
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(avg_composition.keys(), avg_composition.values(), color=['#FF6347', '#4682B4', '#32CD32'])
    
    for bar in ax.patches:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f'{bar.get_height():.2%}', ha='center', color='black', fontsize=12, weight='bold')

    plt.title('Average Baggage Composition Per Flight', fontsize=16, weight='bold')
    plt.ylabel('Proportion of Total Bags', fontsize=12)
    plt.ylim(0, 1)
    plt.savefig('output/2_eda_baggage_composition.png')

    master_df['load_category'] = pd.qcut(master_df['load_factor'], 2, labels=['Low Load', 'High Load'])
    master_df['ssr_category'] = np.where(master_df['ssr_count'] > master_df['ssr_count'].median(), 'High SSR', 'Low SSR')
    delay_by_group = master_df.groupby(['load_category', 'ssr_category'])['departure_delay'].mean().unstack()
    
    delay_by_group.plot(kind='bar', figsize=(10, 7), rot=0, color=['#FFC300', '#C70039'])
    plt.title('Average Delay by Load and SSR Count', fontsize=16, weight='bold')
    plt.xlabel('Passenger Load Category', fontsize=12)
    plt.ylabel('Average Departure Delay (Minutes)', fontsize=12)
    plt.legend(title='SSR Category')
    plt.tight_layout()
    plt.savefig('output/3_eda_ssr_vs_delay.png')

if __name__ == '__main__':
    visualize_eda()

