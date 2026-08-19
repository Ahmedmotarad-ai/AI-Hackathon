# Stage 3: Temporal Pattern / Anomaly Detection Report

## Dataset

- **Subjects:** 15
- **Minute observations:** 471,810
- **Date range:** 2016-03-29 00:00:00 to 2016-05-12 16:20:00

## Detection Method

Stage 3 detects temporal deviations from each person's own established baseline.
The primary comparison is against the individual's median heart rate (personal baseline).
Hourly baselines from Stage 2 are used to provide circadian context, reducing
false positives from normal daily variation.

Each minute observation is scored for:
- Absolute deviation from personal baseline (bpm)
- Relative deviation (%)
- Robust standardized deviation (using Median Absolute Deviation)

Candidate minutes that exceed detection thresholds are grouped into temporal events.
Events are confirmed if they satisfy minimum duration or recurrence requirements.

## Thresholds

**These are prototype engineering/statistical detection parameters, NOT clinical diagnostic thresholds.**

| Parameter | Value | Purpose |
|---|---|---|
| MIN_DURATION_MINUTES | 5 | Minimum consecutive minutes for sustained events |
| MIN_DEVIATION_BPM | 10.0 | Minimum bpm deviation to flag a candidate minute |
| MAD_THRESHOLD | 3.0 | Robust z-score threshold in MAD units |
| SUDDEN_CHANGE_THRESHOLD | 20.0 | Minimum bpm change between consecutive minutes |
| MIN_RECURRING_OCCURRENCES | 3 | Minimum separate days for a recurring pattern |
| RECURRING_GAP_HOURS | 24 | Maximum gap between occurrences |
| VARIABILITY_MULTIPLIER | 2.5 | Multiplier over baseline std for unusual variability |
| MIN_HOURLY_OBSERVATIONS | 30 | Minimum observations in hourly bucket |
| MIN_READINGS_PER_MINUTE | 2 | Minimum readings to trust a minute's observation |

## Results

- **Candidate minutes:** 183,099
- **Candidate events:** 12302
- **Confirmed events:** 12302
- **Clinical queries generated:** 12302

### Event Types

- sustained_elevation: 4917
- sustained_reduction: 4183
- sudden_change: 3170
- recurring_elevation: 15
- recurring_reduction: 15
- unusual_variability: 2

### Duration Statistics

- Mean: 149.9 minutes
- Median: 7.0 minutes
- Min: 3 minutes
- Max: 62394 minutes

### Confidence Distribution

- HIGH: 9380
- MEDIUM: 2922

## Subject-Level Summary

- **2022484408**: many events (1048) (baseline reliability: HIGH)
- **2026352035**: many events (13) (baseline reliability: MEDIUM) [low reliability]
- **2347167796**: many events (996) (baseline reliability: HIGH)
- **4020332650**: many events (589) (baseline reliability: HIGH)
- **4388161847**: many events (934) (baseline reliability: HIGH)
- **4558609924**: many events (911) (baseline reliability: HIGH)
- **5553957443**: many events (486) (baseline reliability: HIGH)
- **5577150313**: many events (1570) (baseline reliability: HIGH)
- **6117666160**: many events (862) (baseline reliability: HIGH)
- **6391747486**: many events (20) (baseline reliability: LOW) [low reliability]
- **6775888955**: many events (358) (baseline reliability: MEDIUM) [low reliability]
- **6962181067**: many events (1312) (baseline reliability: HIGH)
- **7007744171**: many events (927) (baseline reliability: HIGH)
- **8792009665**: many events (794) (baseline reliability: HIGH)
- **8877689391**: many events (1482) (baseline reliability: HIGH)

## Clinical Queries

- **Total queries generated:** 12302

### Example Queries

**[EVT-0001] sustained_elevation:**
> A person with a personal baseline heart rate of 77 bpm experienced a sustained heart rate around 102 bpm for 9 minutes around 7:00, representing approximately a 25 bpm increase from baseline. What clinical guidance exists regarding sustained elevated heart rate, and what contextual factors should be considered?

**[EVT-0002] sustained_reduction:**
> A person with a personal baseline heart rate of 77 bpm experienced a sustained heart rate around 59 bpm for 20 minutes around 8:00, representing approximately a 18 bpm decrease from baseline. What clinical guidance exists regarding sustained lower heart rate, and what contextual factors should be considered?

**[EVT-0003] sustained_reduction:**
> A person with a personal baseline heart rate of 77 bpm experienced a sustained heart rate around 64 bpm for 5 minutes around 8:00, representing approximately a 13 bpm decrease from baseline. What clinical guidance exists regarding sustained lower heart rate, and what contextual factors should be considered?

**[EVT-0004] sustained_elevation:**
> A person with a personal baseline heart rate of 77 bpm experienced a sustained heart rate around 117 bpm for 87 minutes around 9:00, representing approximately a 40 bpm increase from baseline. What clinical guidance exists regarding sustained elevated heart rate, and what contextual factors should be considered?

**[EVT-0005] sustained_elevation:**
> A person with a personal baseline heart rate of 77 bpm experienced a sustained heart rate around 91 bpm for 5 minutes around 10:00, representing approximately a 14 bpm increase from baseline. What clinical guidance exists regarding sustained elevated heart rate, and what contextual factors should be considered?

## Limitations

- Only 15 subjects analyzed
- Fitbit-derived consumer wearable data
- Limited observation period (approximately 1 month)
- No clinical labels or diagnostic ground truth
- No age, gender, or regional context available
- Activity context not integrated (activity data only available at daily resolution)
- Statistical anomaly detection is NOT medical diagnosis
- Subject 2026352035 has only 5 days of data (MEDIUM reliability)

## RAG Integration Readiness

**YES WITH LIMITATIONS**

Stage 3 produces structured anomaly events and corresponding clinical queries
that are ready for downstream RAG consumption. Each event includes:
- Statistical evidence (deviation, duration, confidence)
- Personal baseline context
- Temporal metadata
- A carefully worded clinical query that avoids diagnosis

Limitations for RAG integration:
- Small subject pool (15)
- No activity/rest context to distinguish exercise from resting HR elevation
- No clinical labels for validation
- Detection thresholds are engineering parameters, not clinically validated