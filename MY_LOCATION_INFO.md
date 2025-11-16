# Your Location - Sea State Analysis

## Location Details

**Name:** UK Northeast Coast
**Area:** Hartlepool/Middlesbrough region
**Purpose:** Marine workability analysis for offshore operations

### Coordinates

**Center Point:**
- Latitude: 54.672745°N
- Longitude: -1.03719°E (1.03719°W)

**Bounding Box (with 0.5° buffer):**
- North: 55.172745°
- South: 54.172745°
- West: -1.53719° (1.53719°W)
- East: -0.53719° (0.53719°W)

**Coverage Area:** ~55 km × 55 km (approximately)

### Data Configuration

**Data Source:** ERA5 Reanalysis (ECMWF)
**Years:** 2020-2023 (4 years)
**Resolution:** Hourly data
**Grid Resolution:** ~30 km (0.25°)

**Variables Downloaded:**
- Significant wave height (Hs)
- Peak wave period (Tp)
- Mean wave direction
- 10m wind speed (U and V components)

## How to Run the Analysis

### 1. Install Dependencies (if not done already)

```bash
pip install -r requirements.txt
```

### 2. Run Your Custom Location Script

```bash
python scripts/download_my_location.py
```

**Expected Duration:** 10-30 minutes (depending on CDS server queue)

### 3. What You'll Get

The script will:
1. ✅ Download 4 years of wave and wind data
2. ✅ Extract time series at your center point
3. ✅ Perform quality control
4. ✅ Generate Hs-Tp scatter diagram
5. ✅ Calculate statistics:
   - Mean/max wave heights
   - Mean/max wave periods
   - Wind speed statistics
   - % of time in different sea states
6. ✅ Save processed data:
   - `data/processed/timeseries/UK_NortheastCoast_2020_2023.parquet`
   - `data/processed/scatter_diagrams/UK_NortheastCoast_2020_2023_scatter.parquet`
   - `data/processed/scatter_diagrams/UK_NortheastCoast_2020_2023_scatter.csv` (easy to open)

## Understanding Your Results

### Scatter Diagram
Shows frequency of different wave conditions (Hs vs Tp combinations)
- **Hs (Significant Wave Height):** The average height of the highest 1/3 of waves
- **Tp (Peak Period):** The wave period with the most energy
- **Frequency:** Number of hours each combination occurred

### Wave Height Distribution
You'll see what % of time the waves are below certain heights:
- Hs < 1.5m → Typical "calm" operations
- Hs < 2.5m → Standard crane operations
- Hs < 3.0m → Moderate conditions
- Hs >= 3.0m → Rough conditions

### Typical Conditions (UK Northeast Coast)
Based on historical data, you can expect:
- **Prevailing waves:** Southwest to Northwest
- **Typical Hs:** 1-2m (most common)
- **Typical Tp:** 6-9 seconds
- **Roughest months:** November - February
- **Calmest months:** May - August

## Next Steps After Download

### Phase 2: Workability Analysis
Once you have the sea state data, we can:
1. Define operational limits (e.g., "no crane ops when Hs > 2.5m")
2. Calculate workability percentage
3. Determine best weather windows
4. Compare different seasons/months

### Useful Questions to Answer:
- What % of time can we perform crane operations? (Hs < 2.5m)
- What are the best months for operations?
- What's the expected downtime?
- What's the risk of encountering Hs > 3m during a 30-day project?

## Location Context

Your area (54.67°N, 1.04°W) is:
- **Sea Area:** North Sea, western side
- **Exposure:** Open to North Sea swells
- **Depth:** ~30-50m (approximate)
- **Typical Uses:**
  - Offshore wind farm operations
  - Oil & gas support
  - Survey work
  - Cable laying

## Support Files

- Configuration: `config/locations.yaml`
- Download script: `scripts/download_my_location.py`
- Full project details: `workability_project.md`

---

**Ready?** Run the script and let's analyze your sea conditions!

```bash
python scripts/download_my_location.py
```
