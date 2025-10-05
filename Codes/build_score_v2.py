import pandas as pd
import numpy as np

def build_final_difficulty_score():

    features_df = pd.read_csv('output/cleaned_flight_features_v2.csv')
    
    # Weights assigned as per results gotten in EDA step
    feature_weights = {
        'ssr_score':          0.30,
        'ground_time_score':  0.30,
        'hot_transfer_score': 0.25,
        'transfer_score':     0.10,
        'total_bags_score':   0.05
    }

    all_days_data = []
    features_df['scheduled_departure_date_local'] = pd.to_datetime(features_df['scheduled_departure_date_local'])
    
    for date, day_df in features_df.groupby('scheduled_departure_date_local'):
        day_df['ssr_score'] = day_df['ssr_count'].rank(pct=True) * 100
        day_df['total_bags_score'] = day_df['total_bags'].rank(pct=True) * 100
        day_df['transfer_score'] = day_df['transfer_ratio'].rank(pct=True) * 100
        day_df['hot_transfer_score'] = day_df['hot_transfer_ratio'].rank(pct=True) * 100
        day_df['ground_time_score'] = day_df['ground_time_cushion'].rank(pct=True, ascending=False) * 100
        
        # Calculating the final weighted score
        day_df['DifficultyScore'] = (
            day_df['ssr_score'] * feature_weights['ssr_score'] +
            day_df['ground_time_score'] * feature_weights['ground_time_score'] +
            day_df['hot_transfer_score'] * feature_weights['hot_transfer_score'] +
            day_df['transfer_score'] * feature_weights['transfer_score'] +
            day_df['total_bags_score'] * feature_weights['total_bags_score']
        )
        all_days_data.append(day_df)
    
    scored_df = pd.concat(all_days_data)

    scored_df['scheduled_departure_date_local'] = scored_df['scheduled_departure_date_local'].dt.strftime('%Y-%m-%d')
    quantiles = scored_df.groupby('scheduled_departure_date_local')['DifficultyScore'].quantile([0.55, 0.85]).unstack()
    
    def classify_flight(row):
        date_quantiles = quantiles.loc[row['scheduled_departure_date_local']]
        if row['DifficultyScore'] >= date_quantiles[0.85]: return 'Difficult'
        elif row['DifficultyScore'] >= date_quantiles[0.55]: return 'Medium'
        else: return 'Easy'
            
    scored_df['DifficultyClass'] = scored_df.apply(classify_flight, axis=1)

    output_cols = [
        'company_id', 'flight_number', 'scheduled_departure_date_local', 
        'scheduled_arrival_station_code', 'DifficultyScore', 'DifficultyClass',
        'ssr_count', 'ground_time_cushion', 'hot_transfer_ratio', 'transfer_ratio', 'total_bags'
    ]
    scored_df.sort_values(by=['scheduled_departure_date_local', 'DifficultyScore'], ascending=[True, False], inplace=True)
    
    scored_df[output_cols].to_csv('output/test_ridhayneerajnathoo.csv', index=False, date_format='%Y-%m-%d')

if __name__ == '__main__':
    build_final_difficulty_score()