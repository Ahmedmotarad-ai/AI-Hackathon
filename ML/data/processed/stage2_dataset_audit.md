# Stage 2 Dataset Audit Report

**Audit Date:** 2026-08-19
**Dataset:** Fitabase (Fitbit) data — two collection periods
**Goal:** Determine if Dataset 2 can integrate with Stage 1 Peer Baseline

---

## 1. Dataset Structure

### Folder Overview

| Folder | Period | Files | Description |
|--------|--------|-------|-------------|
| `mturkfitbit_export_3.12.16-4.11.16` | Mar 12 – Apr 11, 2016 | 11 CSVs (+1 checkpoint) | Period 1 (P1) |
| `mturkfitbit_export_4.12.16-5.12.16` | Apr 12 – May 12, 2016 | 18 CSVs (+2 checkpoints) | Period 2 (P2) |

### Complete File Inventory

#### Period 1 (P1) — `mturkfitbit_export_3.12.16-4.11.16`

| File | Size | Rows | Columns | Unique IDs | Purpose |
|------|------|------|---------|------------|---------|
| `dailyActivity_merged.csv` | 50.1 KB | 457 | 15 | 35 | Daily activity summary (steps, distance, active minutes, calories) |
| `heartrate_seconds_merged.csv` | 39.2 MB | 1,154,681 | 3 | 14 | **Second-level heart rate** |
| `hourlyCalories_merged.csv` | 852.2 KB | 24,084 | 3 | 34 | Hourly calorie burn |
| `hourlyIntensities_merged.csv` | 948.9 KB | 24,084 | 4 | 34 | Hourly activity intensity |
| `hourlySteps_merged.csv` | 845.0 KB | 24,084 | 3 | 34 | Hourly step counts |
| `minuteCaloriesNarrow_merged.csv` | 69.1 MB | 1,445,040 | 3 | 34 | Minute-level calorie burn |
| `minuteIntensitiesNarrow_merged.csv` | 48.2 MB | 1,445,040 | 3 | 34 | Minute-level intensity |
| `minuteMETsNarrow_merged.csv` | 49.6 MB | 1,445,040 | 3 | 34 | Minute-level METs |
| `minuteSleep_merged.csv` | 8.9 MB | 198,559 | 4 | 23 | Minute-level sleep states |
| `minuteStepsNarrow_merged.csv` | 48.3 MB | 1,445,040 | 3 | 34 | Minute-level steps |
| `weightLogInfo_merged.csv` | 3.3 KB | 33 | 8 | 11 | Weight, BMI, fat % (manual log) |

#### Period 2 (P2) — `mturkfitbit_export_4.12.16-5.12.16`

| File | Size | Rows | Columns | Unique IDs | Purpose |
|------|------|------|---------|------------|---------|
| `dailyActivity_merged.csv` | 108.7 KB | 940 | 15 | 33 | Daily activity summary |
| `dailyCalories_merged.csv` | 24.5 KB | 940 | 3 | 33 | Daily calorie burn (subset of dailyActivity) |
| `dailyIntensities_merged.csv` | 68.9 KB | 940 | 10 | 33 | Daily intensity (subset of dailyActivity) |
| `dailySteps_merged.csv` | 24.6 KB | 940 | 3 | 33 | Daily steps (subset of dailyActivity) |
| `heartrate_seconds_merged.csv` | 85.4 MB | 2,483,658 | 3 | 14 | **Second-level heart rate** |
| `hourlyCalories_merged.csv` | 782.7 KB | 22,099 | 3 | 33 | Hourly calorie burn |
| `hourlyIntensities_merged.csv` | 877.7 KB | 22,099 | 4 | 33 | Hourly activity intensity |
| `hourlySteps_merged.csv` | 777.9 KB | 22,099 | 3 | 33 | Hourly step counts |
| `minuteCaloriesNarrow_merged.csv` | 63.4 MB | 1,325,580 | 3 | 33 | Minute-level calorie burn |
| `minuteCaloriesWide_merged.csv` | 21.9 MB | 21,645 | 61 | 33 | Hourly wide-format calories (60 min columns) |
| `minuteIntensitiesNarrow_merged.csv` | 44.2 MB | 1,325,580 | 3 | 33 | Minute-level intensity |
| `minuteIntensitiesWide_merged.csv` | 3.2 MB | 21,645 | 61 | 33 | Hourly wide-format intensities |
| `minuteMETsNarrow_merged.csv` | 45.5 MB | 1,325,580 | 3 | 33 | Minute-level METs |
| `minuteSleep_merged.csv` | 8.4 MB | 188,521 | 4 | 24 | Minute-level sleep states |
| `minuteStepsNarrow_merged.csv` | 44.4 MB | 1,325,580 | 3 | 33 | Minute-level steps |
| `minuteStepsWide_merged.csv` | 3.3 MB | 21,645 | 61 | 33 | Hourly wide-format steps |
| `sleepDay_merged.csv` | 17.7 KB | 413 | 5 | 24 | Daily sleep summary |
| `weightLogInfo_merged.csv` | 6.6 KB | 67 | 8 | 8 | Weight, BMI, fat % (manual log) |

### Subjects Across Periods

- **Total unique individuals (both periods):** 35
- **P1 only:** 2 (IDs: 2891001357, 6391747486)
- **P2 only:** 0
- **Both periods:** 33
- **P1 daily activity subjects:** 35
- **P2 daily activity subjects:** 33
- **P1 heartrate subjects:** 14
- **P2 heartrate subjects:** 14 (12 shared with P1)

---

## 2. Patient/Profile Information

### ID Field

| Property | Value |
|----------|-------|
| **ID column name** | `Id` |
| **ID type** | Integer (Fitbit user ID) |
| **Total unique individuals** | 35 |
| **Missing ID percentage** | 0% |
| **IDs repeat across time-series** | Yes (each person has many daily/minute records) |

### Cross-Period ID Overlap

All 33 P2 IDs exist in P1. P1 has 2 additional IDs not in P2. The two periods cover the **same cohort** of people — they are consecutive time windows for the same Fitbit users.

**Important:** These are MTurk (Amazon Mechanical Turk) workers who shared Fitbit data. This is NOT the same population as the 300 Stage 1 patients.

---

## 3. Stage 1 Profile Feature Availability

Stage 1 uses: `age`, `gender`, `height_cm`, `weight_kg`, `BMI`, `region`

| Stage 1 Feature | Dataset 2 Column | File | Available? | Missing % | Notes |
|-----------------|------------------|------|------------|-----------|-------|
| `age` | — | — | **NO** | 100% | No age data anywhere in the dataset |
| `gender` | — | — | **NO** | 100% | No gender/sex data anywhere in the dataset |
| `height_cm` | — | — | **PARTIAL** | ~63% | **Not stored directly.** Can be derived from `WeightKg` and `BMI` via: `height_cm = sqrt(WeightKg / BMI) * 100`. Only 13/35 people have weight+BMI logs. |
| `weight_kg` | `WeightKg` | `weightLogInfo_merged.csv` | **PARTIAL** | 63% | Only 13 of 35 people have weight log entries. Those who log weight do so consistently. |
| `BMI` | `BMI` | `weightLogInfo_merged.csv` | **PARTIAL** | 63% | Only 13 of 35 people. Same coverage as weight. |
| `region` | — | — | **NO** | 100% | No region/location data anywhere in the dataset |

### Key Finding: 3 of 6 Stage 1 features are completely absent (age, gender, region). 2 features (weight, BMI) are available for only 37% of subjects. Height can be derived but only for the same 37%.

---

## 4. Heart Rate Data Audit

### Period 1 Heartrate

| Property | Value |
|----------|-------|
| **File** | `heartrate_seconds_merged.csv` (P1) |
| **Columns** | `Id`, `Time`, `Value` |
| **Total records** | 1,154,681 |
| **Unique subjects** | 14 |
| **Time range** | Mar 29, 2016 – Apr 12, 2016 (~14 days) |
| **Timestamp resolution** | 5 seconds (irregular gaps: 5s, 10s, 15s typical) |
| **HR min** | 36 bpm |
| **HR max** | 185 bpm |
| **HR mean** | 78.3 bpm |
| **HR median** | 75.0 bpm |
| **HR std** | 15.71 bpm |
| **HR missing** | 0% |
| **Duplicates (Id+Time)** | 0 |

### Period 2 Heartrate

| Property | Value |
|----------|-------|
| **File** | `heartrate_seconds_merged.csv` (P2) |
| **Columns** | `Id`, `Time`, `Value` |
| **Total records** | 2,483,658 |
| **Unique subjects** | 14 |
| **Time range** | Apr 12, 2016 – May 21, 2016 (~28 days) |
| **Timestamp resolution** | 5 seconds (irregular gaps: 5s most common, some 10s, 15s) |
| **HR min** | 36 bpm |
| **HR max** | 203 bpm |
| **HR mean** | 77.81 bpm |
| **HR median** | 75.0 bpm |
| **HR std** | 17.98 bpm |
| **HR missing** | 0% |
| **Duplicates (Id+Time)** | 0 |

### Per-Person HR Statistics (P2)

| ID | Records | Days Covered | HR Min | HR Max | HR Mean | HR Median |
|----|---------|--------------|--------|--------|---------|-----------|
| 2022484408 | 154,104 | 28 | 38 | 203 | 80.2 | 76 |
| 2026352035 | 2,490 | 23 | 63 | 125 | 93.8 | 95 |
| 2347167796 | 152,683 | 18 | 49 | 195 | 76.7 | 73 |
| 4020332650 | 285,461 | 28 | 46 | 191 | 82.3 | 83 |
| 4388161847 | 249,748 | 27 | 39 | 180 | 66.1 | 62 |
| 4558609924 | 192,168 | 28 | 44 | 199 | 81.7 | 80 |
| 5553957443 | 255,174 | 28 | 47 | 165 | 68.6 | 64 |
| 5577150313 | 248,560 | 28 | 36 | 174 | 69.6 | 62 |
| 6117666160 | 158,899 | 25 | 52 | 189 | 83.7 | 84 |
| 6775888955 | 32,771 | 25 | 55 | 177 | 92.0 | 91 |
| 6962181067 | 266,326 | 28 | 47 | 184 | 77.7 | 73 |
| 7007744171 | 133,592 | 25 | 54 | 166 | 91.1 | 90 |
| 8792009665 | 122,841 | 23 | 43 | 158 | 72.5 | 70 |
| 8877689391 | 228,841 | 28 | 46 | 180 | 83.6 | 72 |

### HR Data Quality Observations

- **No missing HR values** — all records have a numeric value
- **No duplicate timestamps** — each (Id, Time) pair is unique
- **Timestamp resolution is ~5 seconds** but with irregular gaps (some 10s, 15s, occasionally larger gaps)
- **Not perfectly continuous** — Fitbit sensors record when motion is detected, so there are natural gaps (sleep, inactivity)
- **Best subjects** have 150K-285K records (28 days of frequent recording)
- **Minimum records per person:** 2,490 (ID 2026352035) — still sufficient for baseline computation
- **Subject 2026352035** is an outlier with far fewer records and higher mean HR (93.8)

---

## 5. Integration Feasibility Assessment

### A. Peer Baseline

**Can a Dataset 2 person be mapped into Stage 1's feature space?**

| Feature | Available | Coverage | How |
|---------|-----------|----------|-----|
| age | NO | 0% | Not in dataset at all |
| gender | NO | 0% | Not in dataset at all |
| height_cm | DERIVABLE | 37% (13/35) | From BMI + WeightKg |
| weight_kg | YES | 37% (13/35) | From weightLogInfo |
| BMI | YES | 37% (13/35) | From weightLogInfo |
| region | NO | 0% | Not in dataset at all |

**Result: FULL Peer Baseline mapping is NOT possible.** Missing age, gender, and region means we cannot replicate the Stage 1 KNN similarity model. At most 3 of 6 features can be populated, and only for 37% of subjects.

### B. Personal Baseline

**Does Dataset 2 provide enough HR data per person to compute personal baselines?**

- 14 subjects have heartrate data
- Minimum 2,490 records per subject, maximum 285,461
- All subjects have multiple days of data (18-28 days)
- HR range is physiologically reasonable (36-203 bpm)
- Can compute: resting HR, mean HR, median HR, HR variability, personal baseline

**Result: YES — the 14 HR subjects have excellent data for personal baseline computation.**

### C. Temporal Pattern / Anomaly

**Does Dataset 2 provide enough temporal resolution for anomaly detection?**

- Timestamp resolution: ~5 seconds
- Multiple weeks of data per person
- Can detect: sudden HR spikes, sustained elevation, time-of-day patterns, daily/weekly rhythms
- Can compute: deviation from personal baseline, unusual HR patterns

**Result: YES — the temporal resolution and coverage are well-suited for pattern/anomaly detection.**

---

## 6. Critical Compatibility Check

### Architecture Feasibility

```
Stage 1 Reference Population (300 patients, 6 features)
        ↓
Patient Profiles (age, gender, height, weight, BMI, region)
        ↓
Peer Groups (KNN similarity)
        ↓
Dataset 2 New Person (35 people, only 14 with HR)
        ↓
Build Same Profile Representation  ← BLOCKED: missing age, gender, region
        ↓
Map Person Into Stage 1 Feature Space  ← NOT POSSIBLE with available data
        ↓
Find Similar Reference Patients  ← NOT POSSIBLE
        ↓
Use Dataset 2 Minute-Level HR  ← THIS PART WORKS
        ↓
Personal Baseline  ← THIS PART WORKS
        ↓
Temporal Pattern / Anomaly  ← THIS PART WORKS
```

### Specific Answers

| # | Question | Answer |
|---|----------|--------|
| 1 | Can Dataset 2 people be mapped into Stage 1 profile space? | **NO** — 3 of 6 features completely missing |
| 2 | Which Stage 1 features are available? | `weight_kg` (37%), `BMI` (37%), derivable `height_cm` (37%) |
| 3 | Which Stage 1 features are missing? | `age` (100%), `gender` (100%), `region` (100%) |
| 4 | Can missing features be derived safely? | **NO** — age, gender, region cannot be derived from Fitbit data alone |
| 5 | Is there enough HR data per person? | **YES** — 14 subjects with 2,490–285,461 records across 18–28 days |
| 6 | Is the timestamp information usable? | **YES** — ~5-second resolution, clean timestamps, no duplicates |
| 7 | Which CSVs should actually be used? | `heartrate_seconds_merged.csv`, `weightLogInfo_merged.csv`, `dailyActivity_merged.csv` |
| 8 | Which CSVs are irrelevant? | Wide-format minute files (redundant), hourly files (lower resolution than minute), `minuteSleep` (no HR linkage) |

---

## 7. Recommended File Classification

### Files to USE

| File | Period | Reason |
|------|--------|--------|
| `heartrate_seconds_merged.csv` | P2 | Core HR data — 2.48M records, 14 subjects, 5-second resolution |
| `heartrate_seconds_merged.csv` | P1 | Additional HR data — 1.15M records, 14 subjects, fills earlier time window |
| `weightLogInfo_merged.csv` | P2 | Weight + BMI for 8 people (best coverage) |
| `weightLogInfo_merged.csv` | P1 | Weight + BMI for 11 people |
| `dailyActivity_merged.csv` | P2 | Daily steps, active minutes, calories — useful for context |
| `dailyActivity_merged.csv` | P1 | Same for earlier period |

### Files POSSIBLY USE (backup/supplementary)

| File | Period | Reason |
|------|--------|--------|
| `sleepDay_merged.csv` | P2 | Sleep summary could supplement HR context (24 subjects) |
| `minuteSleep_merged.csv` | P2 | Minute-level sleep states — could help identify sleep vs wake HR |
| `hourlyCalories_merged.csv` | P2 | Lower-resolution calorie data |
| `hourlyIntensities_merged.csv` | P2 | Lower-resolution intensity data |
| `hourlySteps_merged.csv` | P2 | Lower-resolution step data |

### Files DO NOT USE

| File | Period | Reason |
|------|--------|--------|
| `minuteCaloriesWide_merged.csv` | P2 | Redundant — wide format of same data |
| `minuteCaloriesNarrow_merged.csv` | P1/P2 | Redundant — calorie-only (already in dailyActivity) |
| `minuteIntensitiesWide_merged.csv` | P2 | Redundant — wide format |
| `minuteIntensitiesNarrow_merged.csv` | P1/P2 | Redundant — intensity-only |
| `minuteMETsNarrow_merged.csv` | P1/P2 | METs not needed for current architecture |
| `minuteStepsNarrow_merged.csv` | P1/P2 | Steps already in dailyActivity |
| `minuteStepsWide_merged.csv` | P2 | Redundant — wide format |
| `hourlyCalories_merged.csv` | P1 | Redundant with daily/hourly P2 |
| `hourlyIntensities_merged.csv` | P1 | Redundant |
| `hourlySteps_merged.csv` | P1 | Redundant |
| `dailyCalories_merged.csv` | P2 | Subset of dailyActivity |
| `dailyIntensities_merged.csv` | P2 | Subset of dailyActivity |
| `dailySteps_merged.csv` | P2 | Subset of dailyActivity |
| `.ipynb_checkpoints/*` | P1/P2 | Jupyter checkpoint files — ignore |

---

## 8. Integration Feasibility Verdict

### REQUIRES SIGNIFICANT PREPROCESSING

**Why:** The dataset is excellent for HR analysis (personal baseline + temporal anomaly) but CANNOT be directly integrated with Stage 1's peer similarity model due to missing demographic features (age, gender, region).

### Path Forward Options

**Option A: Use Only HR Layer (Recommended for MVP)**
- Skip the peer-similarity mapping entirely for Dataset 2
- Use Dataset 2's own 35 subjects as their own reference population
- Compute personal HR baselines from Dataset 2's HR data
- Detect anomalies relative to each person's own baseline
- Stage 1 peer groups remain a separate, independent reference

**Option B: Derive Proxy Features**
- Use population-level imputation for age/gender (not recommended — unreliable)
- Use region = "unknown" as a catch-all (possible but weakens similarity)
- Height derivable from BMI+Weight for 37% of subjects

**Option C: Obtain Additional Metadata**
- If the original Fitabase dataset includes demographic data in a separate file not provided here, that could fill the gaps
- The original Fitabase/Mturk study likely collected demographics but they are not in these CSV exports

---

## 9. Most Important Finding

> **Can we use Dataset 2's different people and minute-level HR measurements as a second data source while using Stage 1's 300 patients as a reference peer population, without requiring the same people to exist in both datasets?**

**Answer: NOT DIRECTLY — but YES with a modified architecture.**

The fundamental blocker is that Dataset 2 (Fitabase/Fitbit) **does not contain age, gender, or region** — the three categorical/age features that define Stage 1's peer similarity space. Without these, we cannot map Dataset 2 people into Stage 1's feature space, and therefore cannot find "similar" Stage 1 peers for Dataset 2 individuals.

However, Dataset 2's HR data is **excellent**: 14 subjects, 3.6M total HR records across both periods, 5-second resolution, 18–28 days of coverage, no missing values, no duplicates. This is more than sufficient for:
- Computing personal HR baselines per individual
- Detecting temporal HR anomalies (spikes, sustained elevation, abnormal patterns)
- Time-of-day and day-of-week pattern analysis

**The cleanest integration path is a two-layer architecture:**
1. **Stage 1 layer** (unchanged): 300 patients with full profiles → peer groups → reference population
2. **Stage 2 layer** (new): 35 Fitbit subjects with HR time-series → personal baselines → anomaly detection

Stage 2 does NOT need to be mapped into Stage 1's feature space. Instead, Stage 2 operates as an independent HR monitoring pipeline that can run alongside Stage 1's peer-based approach. If demographic data becomes available later, the two layers can be connected.
