# Phase 5E: Integrated Risk Dashboard

Interactive Streamlit dashboard combining all Phase 5 analyses for offshore operations planning.

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r app/requirements.txt
```

### 2. Run Dashboard

```bash
streamlit run app/phase5e_dashboard.py
```

The dashboard will open in your browser at `http://localhost:8501`

## 📊 Dashboard Features

### Pages

1. **🏠 Executive Summary**
   - Key metrics (best/worst months)
   - Quick insights and recommendations
   - Seasonal comparison
   - Module status overview

2. **📅 Monthly Recommendations**
   - Month-by-month detailed analysis
   - Combined score breakdown
   - Radar charts
   - Specific recommendations per month

3. **🌤️ Weather Windows (5D1)**
   - Continuous window probabilities (1-10 days)
   - Interactive duration selector
   - Copula-enhanced calculations
   - Comparison tables

4. **⚠️ Extreme Risk (5D2)**
   - Extreme event encounter probabilities
   - Project duration analysis (7-90 days)
   - Risk categorization (Low/Med/High)
   - Return period insights

5. **🎯 Optimal Scheduling (5D3)**
   - Monthly rankings
   - Combined score (workability + safety + persistence)
   - Month comparison tool
   - Historical window occurrences

6. **📈 Seasonal Analysis (5B)**
   - Seasonal EVA results
   - Kendall's τ by season
   - Directional analysis insights
   - Tidal phase patterns

7. **⏱️ Project Delays (5A)**
   - Historical delay predictions
   - Bootstrap resampling results
   - Monte Carlo uncertainty (P10/P50/P90)

8. **📄 Generate Report**
   - Export functionality (planned)
   - PDF report generation
   - Customizable sections

## 🎯 Use Cases

### For Project Managers
- Identify optimal months for operations
- Estimate project delays with uncertainty
- Plan contingency schedules

### For Operations Teams
- Real-time weather window assessment
- Extreme event risk monitoring
- Day-to-day decision support

### For Commercial/Tendering
- Risk-adjusted pricing
- Seasonal cost variations
- Insurance premium justification

## ⚠️ Important Notes

### Location-Specific
This dashboard analyzes **UK Northeast Coast (2015-2025)** data only.

**Different locations will have:**
- Different extreme patterns
- Different seasonal variations
- Different Hs-Wind-Current dependence structures

**→ Re-analysis required for other sites!**

### Data Requirements

The dashboard expects Phase 5 results in:
```
data/processed/
├── phase5a/
│   └── delay_prediction_results.pkl
├── phase5b1/
│   └── seasonal_eva_results.pkl
├── phase5b2/
│   └── seasonal_copulas.pkl
├── phase5d1/
│   └── weather_window_results.pkl
├── phase5d2/
│   └── extreme_risk_results.pkl
└── phase5d3/
    └── operational_forecast_results.pkl
```

## 🔧 Customization

### Modify Operation Limits
Edit limits in `phase5e_dashboard.py`:
```python
OPERATION_LIMITS = {
    'Crane': {'hs': 2.0, 'wind': 15.0},
    'Diving': {'hs': 2.0, 'current': 0.8},
    'ROV': {'hs': 2.5, 'wind': 18.0}
}
```

### Add New Visualizations
Use Plotly/Matplotlib in respective page functions.

### Change Scoring Weights
Modify in `show_optimal_scheduling()`:
```python
# Current: 40% workability, 30% safety, 30% persistence
combined_score = 0.4 * workability + 0.3 * safety + 0.3 * persistence
```

## 📚 Phase 5 Module Summary

| Module | Purpose | Key Output |
|--------|---------|------------|
| **5A** | Historical Delay Prediction | P50/P90 project durations |
| **5B1** | Seasonal EVA | Return periods by season |
| **5B2** | Seasonal Copulas | Kendall's τ variation |
| **5B3** | Directional Analysis | Wave direction extremes |
| **5B4** | Tidal Phase Analysis | Current by tidal cycle |
| **5C1** | Project Exceedance Risk | Extreme encounter probability |
| **5C2** | Contingency Optimization | Risk-adjusted contingency |
| **5C3** | Multi-Operation Scheduling | Optimal operation sequence |
| **5D1** | Weather Windows | Copula-enhanced probabilities |
| **5D2** | Return Period Windows | Extreme-free periods |
| **5D3** | Operational Forecasting | Combined recommendations |
| **5E** | **Integrated Dashboard** | **All of the above!** |

## 🐛 Troubleshooting

### "Module not found" Error
```bash
pip install streamlit plotly
```

### Dashboard Won't Load
Check that you're in the project root directory:
```bash
cd C:\Work\Project-Offshore-Weather
streamlit run app/phase5e_dashboard.py
```

### Data Not Loading
Ensure Phase 5 notebooks have been run and results saved:
- Run Phase 5D1, 5D2, 5D3 notebooks
- Check `data/processed/` folder

## 📖 Documentation

For detailed methodology, see:
- `notebooks/PHASE5D1_Copula_Weather_Windows.ipynb`
- `notebooks/PHASE5D2_Return_Period_Windows.ipynb`
- `notebooks/PHASE5D3_Operational_Window_Forecasting.ipynb`

## 🎓 Citation

If using this dashboard for research/commercial purposes, please cite the methodology:
- Copula theory for metocean dependence
- Extreme Value Analysis (EVA) for return periods
- Location-specific disclaimers

---

**Built with:** Streamlit, Plotly, Pandas, NumPy, SciPy
**Author:** Phase 5E Development Team
**Version:** 1.0.0
