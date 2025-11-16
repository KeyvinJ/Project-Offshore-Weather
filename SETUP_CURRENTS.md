# 🌊 Ocean Current Data Setup Guide

## Quick Start

You've chosen to add ocean currents! Here's how:

### Step 1: Install Required Package

```bash
pip install openmeteo-requests requests-cache retry-requests
```

OR add to your `requirements.txt`:
```
openmeteo-requests
requests-cache
retry-requests
```

Then: `pip install -r requirements.txt`

### Step 2: Download Current Data (10 years)

```bash
python scripts/download_currents_10_years.py
```

**What this does:**
- Downloads 2015-2025 ocean current data
- From Open-Meteo Marine API (free!)
- Hourly resolution
- ~3-4 hours to complete

**Output:** `data/raw/currents/ocean_currents_YYYY.parquet`

### Step 3: Merge with Wave/Wind Data

```bash
python scripts/merge_currents_with_metocean.py
```

**What this does:**
- Combines hourly currents with 6-hourly waves/wind
- Downsamples currents to 6-hourly (averaging)
- Creates complete metocean dataset

**Output:** `data/processed/timeseries/UK_NortheastCoast_2015_2025_complete.parquet`

### Step 4: Use in Phase 2!

Now your Phase 2 notebooks can use current limits!

---

## What You Get

### Variables in Complete Dataset:

**From Phase 1 (waves/wind):**
- `hs` - Significant wave height (m)
- `tp` - Peak wave period (s)
- `dir` - Wave direction (°)
- `wind_speed` - Wind speed (m/s)
- `wind_u`, `wind_v` - Wind components
- `wind_direction` - Wind direction (°)

**NEW (currents):**
- `current_velocity` - Current speed (m/s)
- `current_speed_knots` - Current speed (knots)
- `current_direction` - Current direction (°)
- `current_u`, `current_v` - Current components

### Data Resolution:
- **Time:** 6-hourly (00:00, 06:00, 12:00, 18:00 UTC)
- **Period:** 2015-2025 (10+ years)
- **Location:** UK Northeast Coast (54.67°N, 1.04°W)

---

## Hourly vs 6-hourly - Decision Explained

### Why we downsample currents to 6-hourly:

✅ **Pros:**
- Matches wave/wind resolution
- Simpler analysis (all variables aligned)
- Smaller dataset, faster processing
- Standard offshore industry practice

❌ **What we lose:**
- Some tidal current detail
- Short-term current variations

### Why it's OK:

1. **6-hourly averages capture main patterns**
   - Tidal cycles are ~12 hours (semi-diurnal)
   - 6-hourly catches high/low tides
   - Smooths out noise

2. **Operations are usually 6-12+ hour windows anyway**
   - Not affected by minute-to-minute variations
   - Care about sustained conditions

3. **Can always go back to hourly if needed**
   - Raw hourly data preserved
   - Easy to re-process if needed

---

## Data Source Info

### Open-Meteo Marine API

**Website:** https://open-meteo.com/en/docs/marine-weather-api

**Advantages:**
- ✅ Free, no quota limits
- ✅ Easy API, no authentication
- ✅ Hourly resolution
- ✅ Global coverage
- ✅ Includes both velocity and direction

**Data Source:**
- Based on NOAA/NCEP ocean models
- Combines multiple ocean reanalysis datasets
- Good accuracy for offshore locations

**Limitations:**
- ~10-25km spatial resolution (coarse for coastal)
- May underestimate strong tidal currents
- Better for open ocean than near-shore

### For UK Northeast Coast:

- Moderate tidal currents (0.5-1.5 knots typical)
- Open-Meteo should capture main patterns
- Good enough for most workability analyses

**If you need higher accuracy:**
- Use dedicated tidal models (FES2014, TPXO)
- UK Hydrographic Office tidal predictions
- Only necessary for critical diving/ROV ops

---

## Typical Current Limits

Use these in Phase 2 workability analysis:

| Operation | Typical Current Limit | Notes |
|-----------|----------------------|-------|
| **Diving** | 0.5-1.0 knots | Most restrictive |
| **ROV operations** | 1.0-1.5 knots | Depends on ROV size |
| **Cable laying** | 1.0-1.5 knots | Affects cable tension |
| **DP vessel positioning** | 1.5-2.5 knots | Depends on vessel |
| **Subsea construction** | 0.5-1.5 knots | Varies by task |
| **Pipeline laying** | 1.0-2.0 knots | Similar to cables |

**Note:** Currents are COMBINED with waves/wind in analysis:
- `workable = (Hs < 1.5m) AND (Wind < 10 m/s) AND (Current < 1.0 knots)`

---

## Troubleshooting

### Error: "openmeteo_requests not found"
```bash
pip install openmeteo-requests requests-cache retry-requests
```

### Download is slow
- Normal! ~10 years of hourly data
- Each year takes ~30-60 seconds
- Total: ~10-20 minutes
- Progress shown year by year

### Some years fail
- Check internet connection
- Re-run script (it will skip successful years)
- Or download failed years individually

### Merge shows missing currents
- Check that download completed successfully
- Verify files exist in `data/raw/currents/`
- Time ranges must overlap with wave/wind data

---

## Summary

**Time investment:**
- Install packages: 1 minute
- Download currents: 20 minutes
- Merge data: 2 minutes
- **Total: ~25 minutes**

**Value:**
- Complete metocean dataset
- Current-limited workability analysis
- Professional-grade offshore planning

**Recommended for:**
- ✅ Diving, ROV, cable operations
- ✅ Complete project analysis
- ✅ Any current-sensitive work

**Optional for:**
- 🟡 General crane/construction (waves dominate)
- 🟡 Wind-limited operations

---

## Next Steps

After completing Steps 1-3 above:

1. ✅ You'll have complete dataset with currents
2. ✅ Phase 2 notebooks will use current criteria
3. ✅ Start Date Optimizer will consider currents
4. ✅ Professional-grade offshore analysis! 🎉

Let's do this! 🚀
