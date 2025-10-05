import pandas as pd
import numpy as np

def create_combined_feature_file():

    #Loading All Data Files ---
    flight_df = pd.read_csv('Dataset/Flight+Level+Data.csv')
    bag_df = pd.read_csv('Dataset/Bag+Level+Data.csv')
    pnr_flight_df = pd.read_csv('Dataset/PNR+Flight+Level+Data.csv')
    pnr_remark_df = pd.read_csv('Dataset/PNR+Remark+Level+Data.csv')

    # Removing duplicate bag tags, keeping the first instance
    bag_df.drop_duplicates(subset=['bag_tag_unique_number'], keep='first', inplace=True)

    # Validating bag tag issue date
    bag_df['bag_tag_issue_date'] = pd.to_datetime(bag_df['bag_tag_issue_date'])
    bag_df['scheduled_departure_date_local'] = pd.to_datetime(bag_df['scheduled_departure_date_local'])
    bag_df = bag_df[bag_df['bag_tag_issue_date'] <= bag_df['scheduled_departure_date_local']]

    #Cleaning and Preparing Core Data
    flight_id_cols = ['company_id', 'flight_number', 'scheduled_departure_date_local']
    flight_df['scheduled_departure_date_local'] = pd.to_datetime(flight_df['scheduled_departure_date_local'])

    # Standardize datetime columns
    for col in ['scheduled_departure_datetime_local', 'actual_departure_datetime_local', 'actual_arrival_datetime_local']:
        flight_df[col] = pd.to_datetime(flight_df[col], errors='coerce')

    # Remove invalid flight records
    flight_df.dropna(subset=['actual_departure_datetime_local'], inplace=True)
    flight_df = flight_df[flight_df['actual_arrival_datetime_local'] >= flight_df['actual_departure_datetime_local']]

    flight_df.dropna(subset=['scheduled_departure_datetime_local'], inplace=True)
    flight_df = flight_df[flight_df['scheduled_arrival_datetime_local'] >= flight_df['scheduled_departure_datetime_local']]
    
    # CRITICAL FIX: Remove flights with invalid (non-positive) scheduled ground time
    flight_df = flight_df[flight_df['scheduled_ground_time_minutes'] > 0]
    
    flight_df['flight_key'] = flight_df[flight_id_cols].astype(str).agg('-'.join, axis=1)
    
    # Remove orphan records from bag and PNR files
    valid_flight_keys = set(flight_df['flight_key'])
    bag_df['flight_key'] = bag_df[flight_id_cols].astype(str).agg('-'.join, axis=1)
    bag_df = bag_df[bag_df['flight_key'].isin(valid_flight_keys)]
    
    pnr_flight_df['flight_key'] = pnr_flight_df[flight_id_cols].astype(str).agg('-'.join, axis=1)
    pnr_flight_df = pnr_flight_df[pnr_flight_df['flight_key'].isin(valid_flight_keys)]

    # Aggregating and merging features
    bag_counts = bag_df.groupby('flight_key')['bag_type'].value_counts().unstack(fill_value=0)
    bag_counts.rename(columns={'Origin': 'origin_bags', 'Transfer': 'transfer_bags', 'Hot Transfer': 'hot_transfer_bags'}, inplace=True)
    bag_counts['total_bags'] = bag_counts.sum(axis=1)
    
    bag_counts['transfer_ratio'] = bag_counts.get('transfer_bags', 0) / bag_counts['total_bags']
    bag_counts['hot_transfer_ratio'] = bag_counts.get('hot_transfer_bags', 0) / bag_counts['total_bags']
    
    # Correctly count passengers
    pax_per_booking = pnr_flight_df.drop_duplicates(subset=['flight_key', 'record_locator'])
    pax_counts = pax_per_booking.groupby('flight_key')['total_pax'].sum().reset_index().rename(columns={'total_pax': 'passenger_count'})
    
    # Correctly count SSRs
    pnr_keys = pnr_flight_df[['record_locator', 'flight_number', 'flight_key']].drop_duplicates()
    ssr_dated = pd.merge(pnr_remark_df, pnr_keys, on=['record_locator', 'flight_number'])
    ssr_counts = ssr_dated.groupby('flight_key').size().reset_index(name='ssr_count')
    
    # Merging all features
    features_df = pd.merge(flight_df, bag_counts, on='flight_key', how='left')
    features_df = pd.merge(features_df, pax_counts, on='flight_key', how='left')
    features_df = pd.merge(features_df, ssr_counts, on='flight_key', how='left')
    
    features_df.fillna(0, inplace=True)
        
    # Engineer final features
    features_df['departure_delay'] = (features_df['actual_departure_datetime_local'] - features_df['scheduled_departure_datetime_local']).dt.total_seconds() / 60
    features_df['ground_time_cushion'] = features_df['scheduled_ground_time_minutes'] - features_df['minimum_turn_minutes']
    features_df['load_factor'] = (features_df['passenger_count'] / features_df['total_seats']).clip(upper=1.0)

    features_df.to_csv('output/cleaned_flight_features_v2.csv', index=False)

if __name__ == '__main__':
    create_combined_feature_file()

