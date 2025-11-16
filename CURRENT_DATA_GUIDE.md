# 🌊 Ocean Current Data Guide

## Quick Decision Tree

**Do you need ocean currents?**

### ✅ YES - You NEED currents if:
- Diving operations (typical limit: 0.5-1.0 knots)
- ROV operations (typical limit: 1.0-1.5 knots)
- Cable/pipeline laying (typical limit: 1.0-2.0 knots)
- Subsea construction
- DP vessel operations with tight positioning

### 🟡 MAYBE - Currents are helpful but not critical:
- General DP operations
- Towing operations
- Mooring analysis
- Drift calculations

### ❌ NO - Waves/Wind are sufficient:
- Crane operations (wave height is limiting factor)
- Jack-up operations (wave height is limiting)
- General construction work (wind is limiting)
- Personnel transfer (waves/wind dominate)
- Survey work (waves/wind dominate)

---

## ERA5 Ocean Current Data

### What's Available?

ERA5 provides ocean currents through:

**Dataset:** `reanalysis-era5-single-levels`
**Ocean Variables:**
- `u_current` - Eastward ocean current (m/s)
- `v_current` - Northward ocean current (m/s)

**OR**

**Dataset:** `reanalysis-era5-ocean` (if available)
- More comprehensive ocean data
- Better vertical resolution

### ⚠️ Important Notes:

1. **Spatial Resolution:** ERA5 ocean currents are ~0.25° (~25-30 km)
   - This is COARSE for coastal/tidal currents
   - Better for offshore open ocean

2. **Tidal Currents:** ERA5 does NOT explicitly model tides
   - Tides are implicit in the ocean model
   - For strong tidal areas (UK coast!), may underestimate currents

3. **North Sea Specifics:**
   - UK Northeast Coast has significant tidal currents (0.5-2 knots typical)
   - ERA5 may not fully capture tidal variability
   - Consider if you need dedicated tidal model (TPXO, FES2014)

---

## How to Download Ocean Currents from ERA5

### Step 1: Check CDS API Access

You already have access! (Same credentials as waves/wind)

### Step 2: Modify Download Script

```python
# In scripts/download_ocean_currents.py

import cdsapi
from pathlib import Path

client = cdsapi.Client()

# Your location
center_lat = 54.672745
center_lon = -1.03719
buffer = 0.5

bbox = [
    center_lat + buffer,  # North
    center_lon - buffer,  # West
    center_lat - buffer,  # South
    center_lon + buffer   # East
]

# Download ocean currents
for year in range(2015, 2026):
    request = {
        'product_type': 'reanalysis',
        'format': 'netcdf',
        'variable': [
            'eastward_sea_water_velocity',   # u_current
            'northward_sea_water_velocity',  # v_current
        ],
        'year': str(year),
        'month': [f'{m:02d}' for m in range(1, 13)],
        'day': [f'{d:02d}' for d in range(1, 32)],
        'time': ['00:00', '06:00', '12:00', '18:00'],  # 6-hourly
        'area': bbox,
    }

    output = f'data/raw/era5/ocean_currents_{year}.nc'

    print(f"Downloading {year}...")
    client.retrieve('reanalysis-era5-single-levels', request, output)
```

### Step 3: Parse and Add to Dataset

Similar to wave/wind parsing, extract current speed:
```python
current_speed = sqrt(u_current² + v_current²)
```

---

## Alternative: Tidal Current Models

For **UK coastal waters**, consider dedicated tidal models:

### Option 1: UK Hydrographic Office
- UK tidal prediction services
- Very accurate for UK waters
- Commercial data

### Option 2: Global Tidal Models
- **FES2014** (Finite Element Solution)
- **TPXO9** (OSU Tidal Prediction)
- Free for research
- Better tidal current representation

### Option 3: CMEMS (Copernicus Marine)
- Higher resolution ocean models
- Includes tides
- Free registration

---

## My Recommendation

### For UK Northeast Coast (Your Location):

**Short Answer:** Waves + Wind are probably sufficient for most operations

**Why?**
1. **Your location** (Hartlepool/Middlesbrough):
   - Tidal currents exist but moderate (~0.5-1.5 knots typical)
   - Waves (2-4m in winter) are usually the limiting factor
   - Wind (15-20 m/s events) often limits before currents

2. **Typical operations:**
   - If doing crane/jack-up/construction → Waves dominate
   - If doing diving/ROV/cables → You NEED currents

3. **Data quality:**
   - ERA5 ocean currents are coarse for your coastal location
   - Tidal currents may be underestimated
   - For critical current-limited ops, use dedicated tidal model

### Three-Stage Approach (Recommended):

**Stage 1: NOW**
- Complete Phase 2 with waves + wind
- This covers 80% of operations
- Fast, complete analysis ready soon

**Stage 2: IF NEEDED**
- Download ERA5 ocean currents
- Add as "Phase 1B - Ocean Currents"
- Enhance Phase 2 with current criteria

**Stage 3: IF CRITICAL**
- Get dedicated tidal current data (FES2014, CMEMS)
- For diving/ROV/cable operations
- Most accurate for your coastal location

---

## Workability Impact

### With vs Without Currents:

**Crane Operations:**
- Current limit: Usually not applied
- Impact: **0%** - Currents don't change workability

**Diving Operations:**
- Current limit: 0.5-1.0 knots typical
- Impact: **10-30%** - Can significantly reduce workability
- **Recommendation:** NEED currents!

**Cable Laying:**
- Current limit: 1.0-1.5 knots typical
- Impact: **5-15%** - Moderate impact
- **Recommendation:** Should include currents

**Jack-up Operations:**
- Current limit: Sometimes applied (1-2 knots)
- Impact: **<5%** - Usually not limiting factor
- **Recommendation:** Waves/wind sufficient

---

## Decision Matrix

| Operation Type | Current Critical? | Proceed Without? | Add Currents? |
|----------------|-------------------|------------------|---------------|
| Crane Heavy Lift | ❌ No | ✅ Yes | Later if needed |
| Jack-up Operations | 🟡 Sometimes | ✅ Yes | Later if needed |
| Diving/ROV | ✅ YES | ❌ No | Add now! |
| Cable Laying | ✅ YES | ❌ No | Add now! |
| DP Vessel General | 🟡 Helpful | ✅ Yes | Stage 2 |
| Survey Work | ❌ No | ✅ Yes | Not needed |
| Personnel Transfer | ❌ No | ✅ Yes | Not needed |

---

## Bottom Line

**Ask yourself:** What operations am I analyzing?

- **Crane/Construction/General:** → Proceed to Phase 2! Add currents later if needed
- **Diving/ROV/Cables:** → Add currents first (Phase 1B), then Phase 2
- **Mixed operations:** → Phase 2 now, enhance with currents after

**My recommendation:**
1. Tell me what operations you're focused on
2. We decide together: proceed or add currents
3. Either way, you'll have a complete analysis!

---

## Time Investment

**Download ocean currents:** ~3-4 hours (same as waves/wind)
**Parse and integrate:** ~1 hour
**Update Phase 2 notebook:** ~2 hours

**Total:** Half a day to add currents

**Is it worth it?** Depends on your operations! 🎯
