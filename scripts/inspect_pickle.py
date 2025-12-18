import pickle
import os

file_path = 'data/processed/phase5b1/seasonal_eva_results.pkl'

try:
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    
    print("Checking 'seasonal_eva_results' -> 'hs' -> 'Winter':")
    if 'seasonal_eva_results' in data and 'hs' in data['seasonal_eva_results']:
        winter_data = data['seasonal_eva_results']['hs']['Winter']
        print(winter_data)
        print(f"Keys: {winter_data.keys()}")
        
except Exception as e:
    print(f"Error: {e}")
