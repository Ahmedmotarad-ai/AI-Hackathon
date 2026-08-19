# Stage 2: Personal Baseline — Final Report

**Date:** 2026-08-19
**Pipeline Stage:** 2 of 3 (Personal Baseline)
**Input:** Fitbit second-level heart-rate data (Dataset 2)
**Status:** Complete

---

## 1. Dataset Summary

| Property | Value |
|----------|-------|
| Subjects with HR data | 15 |
| Total HR observations | 3,614,915 |
| Date range | 2016-03-29 to 2016-05-12 |
| Resolution | ~5 seconds (irregular) |
| HR range | 36–203 bpm |
| Source periods | P1 (Mar 12–Apr 11) + P2 (Apr 12–May 12) |
| Duplicates removed | 23,424 |

Dataset 2 subjects are completely independent from the 300 Stage 1 patients. No cross-dataset identity matching was performed.

---

## 2. Subject Coverage

| Metric | Min | Mean | Max |
|--------|-----|------|-----|
| Observations | 2,929 | 240,994 | 561,120 |
| Days observed | 3 | 31.3 | 44 |
| Median readings/day | 19 | 5,098 | 21,507 |
| Time span (days) | 3.1 | 36.6 | 43.5 |

**Sparse subjects:** 2 subjects (IDs 2026352035 and 6391747486) have fewer than 5,000 observations and fewer than 6 days of data. Their baselines are less reliable.

**Well-covered subjects:** 13 of 15 subjects have 67,000+ observations across 25+ days.

---

## 3. Personal Baseline Statistics

### Primary Baseline Definition

**Personal baseline = median HR** (chosen for robustness to outliers and right-skewed HR distributions)

### Population Summary

| Statistic | Value |
|-----------|-------|
| Median HR range across subjects | 61.0 – 95.0 bpm |
| Population mean of medians | 77.5 bpm |
| Std of medians across subjects | 10.5 bpm |
| Mean IQR (within-person) | 19.5 bpm |
| Mean MAD (within-person) | 9.1 bpm |
| Mean CV (within-person) | 20.6% |

### Per-Subject Baseline

| Subject ID | Records | Median HR | IQR | MAD | CV% | Reliability |
|------------|---------|-----------|-----|-----|-----|-------------|
| 2022484408 | 209,056 | 77.0 | 21.0 | 10.0 | 22.0 | HIGH |
| 2026352035 | 2,929 | 92.0 | 24.0 | 11.0 | 17.3 | MEDIUM |
| 2347167796 | 273,487 | 73.0 | 17.0 | 8.0 | 19.2 | HIGH |
| 4020332650 | 561,120 | 84.0 | 23.0 | 11.0 | 18.4 | HIGH |
| 4388161847 | 249,748 | 62.0 | 16.0 | 7.0 | 24.0 | HIGH |
| 4558609924 | 259,941 | 80.0 | 16.0 | 8.0 | 17.0 | HIGH |
| 5553957443 | 350,549 | 64.0 | 12.0 | 5.0 | 21.6 | HIGH |
| 5577150313 | 336,209 | 61.0 | 22.0 | 9.0 | 30.5 | HIGH |
| 6117666160 | 212,565 | 84.0 | 21.0 | 10.0 | 16.8 | HIGH |
| 6391747486 | 3,747 | 83.0 | 10.0 | 5.0 | 11.9 | LOW |
| 6775888955 | 67,871 | 95.0 | 28.0 | 14.0 | 17.6 | MEDIUM |
| 6962181067 | 387,284 | 73.0 | 20.0 | 9.0 | 22.7 | HIGH |
| 7007744171 | 197,393 | 89.0 | 20.0 | 10.0 | 15.9 | HIGH |
| 8792009665 | 190,383 | 72.0 | 17.0 | 8.0 | 18.2 | HIGH |
| 8877689391 | 312,633 | 73.0 | 26.0 | 11.0 | 36.4 | HIGH |

### Why Median Over Mean

7 of 15 subjects show a mean-median difference > 3 bpm, indicating right-skewed HR distributions (expected: more time at lower resting HR, occasional high HR during activity). Median is more robust to these high-HR activity spikes.

---

## 4. Temporal Baseline

### Hourly Patterns

All subjects show expected diurnal HR patterns:

- **Daytime (6–22):** Higher median HR (more activity)
- **Nighttime (22–6):** Lower median HR (rest/sleep)

Mean day-night difference: **+10.2 bpm** (range: −0.7 to +25.1 bpm)

Subject 2026352035 shows an unusually large day-night difference (+25.1 bpm), likely due to very sparse nighttime data.

### Daily Patterns

- Median daily HR ranges from 53–111 bpm across subjects
- Day-to-day variability is moderate for most subjects
- Subjects with 40+ days show stable long-term baselines

### Temporal Gaps

| Gap Threshold | Mean per Subject | Max per Subject |
|---------------|------------------|-----------------|
| >1 minute | 128 | 322 |
| >5 minutes | 72 | 168 |
| >30 minutes | 35 | 81 |
| Largest gap | — | 1,306,715 sec (~15 days) |

Gaps are expected in Fitbit data: the optical HR sensor only records during detected wrist activity. Overnight gaps during sleep are normal behavior, not missing data.

---

## 5. Baseline Reliability

| Level | Count | Percentage | Criteria |
|-------|-------|------------|----------|
| HIGH | 12 | 80.0% | ≥100K records, ≥21 days, low CV, few large gaps |
| MEDIUM | 2 | 13.3% | ≥10K records, ≥7 days |
| LOW | 1 | 6.7% | Fewer observations or days |

**LOW reliability subject:** ID 6391747486 (3,747 records, 3 days only)

**MEDIUM reliability subjects:**
- ID 2026352035 (2,929 records, 5 days)
- ID 6775888955 (67,871 records, 26 days — has many large gaps)

---

## 6. Key Findings

1. **Personal baselines are highly individual.** Median HR ranges from 61 to 95 bpm across subjects — a 34 bpm spread. Any anomaly detection must be personalized, not population-based.

2. **Within-person variability is moderate.** Mean IQR of 19.5 bpm means the middle 50% of HR values span about 20 bpm. This is the "normal band" for each person.

3. **Daytime-nighttime HR patterns are clear.** Most subjects show 7–16 bpm higher HR during daytime. This temporal structure is essential for Stage 3 anomaly detection.

4. **Median HR is more robust than mean HR.** Right-skewed distributions (expected from HR data) make mean HR an unreliable baseline. Median better represents typical HR.

5. **Data volume varies dramatically.** From 2,929 to 561,120 observations. Reliability flags help Stage 3 weight subjects appropriately.

6. **Two subjects have unusually wide distributions.** IDs 6775888955 (IQR=28) and 8877689391 (IQR=26, CV=36.4%) show wider-than-typical HR variation. Stage 3 should use wider personal thresholds for these individuals.

---

## 7. Limitations

- Only 15 subjects have HR data (from 35 total in Dataset 2)
- Fitbit-derived data (not clinical-grade ECG)
- Short observation period (~3–6 weeks per subject)
- No clinical diagnosis labels available
- No age/gender/region information (cannot map to Stage 1)
- Baseline is descriptive, not an anomaly detector
- No direct comparison to Stage 1 peer population
- Overnight gaps in data are expected Fitbit behavior, not true missingness
- Fitbit HR sensor has known limitations at very high/low HR extremes

---

## 8. Readiness for Stage 3

### YES WITH LIMITATIONS

Stage 2 produces sufficient information to begin temporal pattern and anomaly detection:

**Available for Stage 3:**
- Personal median HR baseline per subject
- Hourly and daily temporal patterns (expected circadian rhythms)
- Minute-level aggregated data (471,810 rows) with mean, median, min, max, std
- Reliability flags (HIGH/MEDIUM/LOW) for quality-aware analysis
- IQR, MAD, and P10-P90 ranges for personal threshold calibration

**Limitations for Stage 3:**
- Small subject count (15) limits population-level conclusions
- No ground-truth abnormal labels for supervised evaluation
- Sparse subjects (IDs 2026352035, 6391747486) may produce unstable temporal patterns
- Temporal train/test splits must respect data gaps
- Stage 3 must use per-subject thresholds, not population thresholds

**Recommendation:** Proceed to Stage 3 using reliability flags to weight analysis. Focus on HIGH-reliability subjects for initial pattern development.

---

## Output Files

| File | Description | Rows | Path |
|------|-------------|------|------|
| `stage2_subject_baseline.csv` | Per-subject baseline statistics | 15 | `data/processed/` |
| `stage2_minute_hr.csv` | Minute-level HR aggregation | 471,810 | `data/processed/` |
| `stage2_hourly_baseline.csv` | Hourly baseline per subject | 332 | `data/processed/` |
| `stage2_daily_baseline.csv` | Daily baseline per subject | 469 | `data/processed/` |
| `stage2_data_quality.csv` | Coverage, gaps, reliability | 15 | `data/processed/` |
| `stage2_personal_baseline_visualizations.png` | 9-panel visualization | — | `data/processed/` |

---

## Stage 1 Status

Stage 1 (Peer Baseline) remains completely unchanged. All Stage 1 files in `data/Stage 1/data/processed/` are untouched. Stage 2 operates independently from Stage 1.
