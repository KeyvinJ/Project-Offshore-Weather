import pickle

with open('data/processed/phase4b/copula_parameters.pkl', 'rb') as f:
    data = pickle.load(f)

print("="*80)
print("COPULA PARAMETERS STRUCTURE")
print("="*80)

print("\nTop-level keys:")
for key in data.keys():
    print(f"  - {key}")

print("\n\nhs_wind dictionary:")
print("-"*80)
for key, value in data['hs_wind'].items():
    if isinstance(value, dict):
        print(f"\n{key}:")
        for subkey in value.keys():
            print(f"    - {subkey}")
    else:
        print(f"{key}: {value}")
