import pickle
import os

try:
    # Load the file created in Phase 5B2
    path = 'data/processed/phase5b2/seasonal_copulas.pkl'
    with open(path, 'rb') as f:
        data = pickle.load(f)
    
    print("✅ VERIFICATION OF DATA SOURCE:")
    print(f"Loading from: {path}")
    
    print("\n--- Stored Kendall's Tau Values (Hs-Wind) ---")
    for season, stats in data['seasonal_dependence'].items():
        tau = stats['hs_wind']['kendall_tau']
        print(f"{season:6s}: {tau:.6f}")
        
except FileNotFoundError:
    print("❌ Error: Phase 5B2 results not found. Please ensure PHASE5B2 has been run.")
except Exception as e:
    print(f"❌ Error: {e}")

