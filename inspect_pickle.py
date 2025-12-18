import pickle
import pprint

print("="*80)
print("PHASE 4A: EVA DISTRIBUTIONS")
print("="*80)
with open('data/processed/phase4a/eva_distributions.pkl', 'rb') as f:
    eva_4a = pickle.load(f)
    print("Keys:", list(eva_4a.keys()))
    pprint.pprint(eva_4a, depth=3)

print("\n" + "="*80)
print("PHASE 4B: COPULA PARAMETERS")
print("="*80)
with open('data/processed/phase4b/copula_parameters.pkl', 'rb') as f:
    copula_4b = pickle.load(f)
    print("Keys:", list(copula_4b.keys()))
    pprint.pprint(copula_4b, depth=3)
