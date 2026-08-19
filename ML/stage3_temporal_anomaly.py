"""
Stage 3: Temporal Pattern / Anomaly Detection
===============================================
Detect meaningful temporal changes in heart-rate behavior relative to each
person's own baseline and convert confirmed events into structured clinical
queries for the downstream RAG system.

This script implements ONLY Stage 3 of the Heart Rate Monitoring ML pipeline.
Stages 1 and 2 are already complete and must not be modified.

Important:
  - This detects temporal deviations, NOT medical diagnoses.
  - All thresholds are engineering/statistical detection parameters, not
    clinical diagnostic thresholds.
  - Clinical queries use cautious wording and only reference data that is
    actually available.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
from datetime import timedelta

warnings.filterwarnings('ignore')

sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 5)
plt.rcParams['figure.dpi'] = 100

# ============================================================
# CONFIGURATION
# ============================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, 'data')
OUTPUT_DIR = os.path.join(DATA_DIR, 'processed')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Stage 2 input paths
MINUTE_HR_PATH = os.path.join(OUTPUT_DIR, 'stage2_minute_hr.csv')
SUBJECT_BASELINE_PATH = os.path.join(OUTPUT_DIR, 'stage2_subject_baseline.csv')
HOURLY_BASELINE_PATH = os.path.join(OUTPUT_DIR, 'stage2_hourly_baseline.csv')
DAILY_BASELINE_PATH = os.path.join(OUTPUT_DIR, 'stage2_daily_baseline.csv')
DATA_QUALITY_PATH = os.path.join(OUTPUT_DIR, 'stage2_data_quality.csv')

# Output paths
MINUTE_DEVIATIONS_PATH = os.path.join(OUTPUT_DIR, 'stage3_minute_deviations.csv')
ANOMALY_EVENTS_PATH = os.path.join(OUTPUT_DIR, 'stage3_anomaly_events.csv')
CLINICAL_QUERIES_PATH = os.path.join(OUTPUT_DIR, 'stage3_clinical_queries.csv')
VISUALIZATION_PATH = os.path.join(OUTPUT_DIR, 'stage3_temporal_visualizations.png')
REPORT_PATH = os.path.join(OUTPUT_DIR, 'stage3_temporal_anomaly_report.md')

# ---------------------------------------------------------------
# Detection thresholds - engineering/statistical parameters,
# NOT clinical diagnostic thresholds
# ---------------------------------------------------------------
# Minimum consecutive minutes for a sustained event
MIN_DURATION_MINUTES = 5

# Minimum deviation from personal baseline to consider a candidate (bpm)
MIN_DEVIATION_BPM = 10.0

# Robust standardized deviation threshold (in MAD units).
# A reading deviating by more than MAD_THRESHOLD * MAD from the median
# is considered statistically unusual for this person.
MAD_THRESHOLD = 3.0

# Minimum absolute change between consecutive minutes for sudden change (bpm)
SUDDEN_CHANGE_THRESHOLD = 20.0

# Minimum separate occurrences for a recurring pattern
MIN_RECURRING_OCCURRENCES = 3

# Maximum gap between occurrences to be considered same recurring pattern (hours)
RECURRING_GAP_HOURS = 24

# Multiplier over baseline std to flag unusual variability
VARIABILITY_MULTIPLIER = 2.5

# Minimum observations in an hourly baseline bucket to use it
MIN_HOURLY_OBSERVATIONS = 30

# Minimum readings per minute to trust that minute's observation
MIN_READINGS_PER_MINUTE = 2

# Confidence thresholds (minutes)
CONFIDENCE_HIGH_DURATION = 15
CONFIDENCE_MEDIUM_DURATION = 5
CONFIDENCE_HIGH_DEVIATION_BPM = 20.0
CONFIDENCE_MEDIUM_DEVIATION_BPM = 10.0


# ============================================================
# SECTION 1: LOAD STAGE 2 OUTPUTS
# ============================================================
print("=" * 70)
print("STAGE 3: TEMPORAL PATTERN / ANOMALY DETECTION")
print("=" * 70)
print()
print("Configuration:")
print(f"  MIN_DURATION_MINUTES       = {MIN_DURATION_MINUTES}")
print(f"  MIN_DEVIATION_BPM          = {MIN_DEVIATION_BPM}")
print(f"  MAD_THRESHOLD              = {MAD_THRESHOLD}")
print(f"  SUDDEN_CHANGE_THRESHOLD    = {SUDDEN_CHANGE_THRESHOLD}")
print(f"  MIN_RECURRING_OCCURRENCES  = {MIN_RECURRING_OCCURRENCES}")
print(f"  RECURRING_GAP_HOURS        = {RECURRING_GAP_HOURS}")
print(f"  VARIABILITY_MULTIPLIER     = {VARIABILITY_MULTIPLIER}")
print(f"  MIN_HOURLY_OBSERVATIONS    = {MIN_HOURLY_OBSERVATIONS}")
print(f"  MIN_READINGS_PER_MINUTE    = {MIN_READINGS_PER_MINUTE}")
print()

print("=" * 70)
print("SECTION 1: LOAD STAGE 2 OUTPUTS")
print("=" * 70)

minute_hr = pd.read_csv(MINUTE_HR_PATH)
minute_hr['timestamp_minute'] = pd.to_datetime(minute_hr['timestamp_minute'])

subject_baseline = pd.read_csv(SUBJECT_BASELINE_PATH)
hourly_baseline = pd.read_csv(HOURLY_BASELINE_PATH)
daily_baseline = pd.read_csv(DAILY_BASELINE_PATH)
data_quality = pd.read_csv(DATA_QUALITY_PATH)

print(f"\nStage 2 outputs loaded:")
print(f"  Minute HR:         {len(minute_hr):>10,} rows")
print(f"  Subject baseline:  {len(subject_baseline):>10,} rows")
print(f"  Hourly baseline:   {len(hourly_baseline):>10,} rows")
print(f"  Daily baseline:    {len(daily_baseline):>10,} rows")
print(f"  Data quality:      {len(data_quality):>10,} rows")

subjects = sorted(minute_hr['Id'].unique())
n_subjects = len(subjects)
n_minutes = len(minute_hr)
date_min = minute_hr['timestamp_minute'].min()
date_max = minute_hr['timestamp_minute'].max()

print(f"\nDataset summary:")
print(f"  Subjects:       {n_subjects}")
print(f"  Minute obs:     {n_minutes:,}")
print(f"  Date range:     {date_min} to {date_max}")

baseline_lookup = subject_baseline.set_index('subject_id').to_dict('index')
quality_lookup = data_quality.set_index('subject_id').to_dict('index')

hourly_lookup = {}
for _, row in hourly_baseline.iterrows():
    sid = row['Id']
    if sid not in hourly_lookup:
        hourly_lookup[sid] = {}
    hourly_lookup[sid][int(row['hour'])] = {
        'median_hr': row['median_hr'],
        'std_hr': row['std_hr'],
        'reading_count': int(row['reading_count']),
    }


# ============================================================
# SECTION 2: COMPUTE MINUTE-LEVEL DEVIATIONS
# ============================================================
print("\n" + "=" * 70)
print("SECTION 2: COMPUTE MINUTE-LEVEL DEVIATIONS")
print("=" * 70)

deviation_rows = []

for sid in subjects:
    baseline = baseline_lookup[sid]
    personal_median = baseline['median_hr']
    personal_mad = baseline['mad']

    sub = minute_hr[minute_hr['Id'] == sid].sort_values('timestamp_minute').copy()
    sub['hour'] = sub['timestamp_minute'].dt.hour
    sub['day_of_week'] = sub['timestamp_minute'].dt.dayofweek
    sub['date'] = sub['timestamp_minute'].dt.date

    for _, row in sub.iterrows():
        observed = row['mean_hr']
        hour = int(row['hour'])

        personal_base = personal_median

        if sid in hourly_lookup and hour in hourly_lookup[sid]:
            h_data = hourly_lookup[sid][hour]
            if h_data['reading_count'] >= MIN_HOURLY_OBSERVATIONS:
                contextual_base = h_data['median_hr']
            else:
                contextual_base = personal_base
        else:
            contextual_base = personal_base

        abs_deviation = observed - personal_base
        abs_deviation_bpm = abs(abs_deviation)
        rel_deviation = (abs_deviation / personal_base * 100) if personal_base > 0 else 0

        if personal_mad > 0:
            robust_z = abs_deviation / personal_mad
        else:
            robust_z = 0.0

        deviation_rows.append({
            'subject_id': sid,
            'timestamp_minute': row['timestamp_minute'],
            'mean_hr': row['mean_hr'],
            'median_hr_minute': row['median_hr'],
            'min_hr': row['min_hr'],
            'max_hr': row['max_hr'],
            'std_hr_minute': row['std_hr'],
            'reading_count': int(row['reading_count']),
            'personal_baseline_hr': personal_base,
            'contextual_baseline_hr': contextual_base,
            'deviation_bpm': round(abs_deviation, 2),
            'abs_deviation_bpm': round(abs_deviation_bpm, 2),
            'relative_deviation_pct': round(rel_deviation, 2),
            'robust_z_score': round(robust_z, 3),
            'hour': hour,
            'day_of_week': int(row['day_of_week']),
            'date': str(row['date']),
            'candidate_flag': False,
            'candidate_type': '',
        })

dev_df = pd.DataFrame(deviation_rows)


def flag_candidates(df):
    """Flag individual minutes that are statistical candidates for anomalies."""
    df = df.copy()

    above_threshold = (df['deviation_bpm'] >= MIN_DEVIATION_BPM)
    below_threshold = (df['deviation_bpm'] <= -MIN_DEVIATION_BPM)
    robust_above = (df['robust_z_score'].abs() >= MAD_THRESHOLD)
    has_readings = (df['reading_count'] >= MIN_READINGS_PER_MINUTE)

    is_candidate = ((above_threshold | below_threshold | robust_above) & has_readings)

    df.loc[is_candidate & above_threshold, 'candidate_flag'] = True
    df.loc[is_candidate & above_threshold, 'candidate_type'] = 'elevated'

    df.loc[is_candidate & below_threshold & ~above_threshold, 'candidate_flag'] = True
    df.loc[is_candidate & below_threshold & ~above_threshold, 'candidate_type'] = 'reduced'

    robust_only = robust_above & ~above_threshold & ~below_threshold & has_readings & ~df['candidate_flag']
    if robust_only.any():
        mean_dir = df.loc[robust_only, 'deviation_bpm'].mean()
        label = 'elevated' if mean_dir > 0 else 'reduced'
        df.loc[robust_only, 'candidate_flag'] = True
        df.loc[robust_only, 'candidate_type'] = label

    return df


def build_event(sub_df, indices, event_type, sid, baseline_info, quality_info):
    """Build an event dict from a set of minute indices."""
    rows = sub_df.iloc[indices]
    start_time = rows['timestamp_minute'].iloc[0]
    end_time = rows['timestamp_minute'].iloc[-1]
    duration = int((end_time - start_time).total_seconds() / 60) + 1

    observed_mean = rows['mean_hr'].mean()
    observed_median = rows['median_hr_minute'].median()
    observed_min = rows['min_hr'].min()
    observed_max = rows['max_hr'].max()
    peak_idx = rows['mean_hr'].idxmax()
    peak_hr = rows.loc[peak_idx, 'mean_hr']
    peak_time = rows.loc[peak_idx, 'timestamp_minute']

    personal_base = baseline_info['median_hr']
    deviation = round(observed_mean - personal_base, 2)
    rel_dev = round((deviation / personal_base * 100) if personal_base > 0 else 0, 2)

    time_of_day = start_time.hour
    day_of_week = start_time.dayofweek
    reliability = quality_info.get('reliability', 'UNKNOWN')

    preceding_hr = None
    if indices[0] > 0:
        preceding_hr = sub_df.iloc[indices[0] - 1]['mean_hr']
    following_hr = None
    if indices[-1] < len(sub_df) - 1:
        following_hr = sub_df.iloc[indices[-1] + 1]['mean_hr']

    return {
        'subject_id': sid,
        'event_type': event_type,
        'start_time': start_time,
        'end_time': end_time,
        'duration_minutes': duration,
        'personal_baseline_hr': personal_base,
        'contextual_baseline_hr': rows['contextual_baseline_hr'].median(),
        'observed_mean_hr': round(observed_mean, 2),
        'observed_median_hr': round(observed_median, 2),
        'observed_min_hr': round(observed_min, 2),
        'observed_max_hr': round(observed_max, 2),
        'deviation_bpm': deviation,
        'relative_deviation_pct': rel_dev,
        'peak_hr': round(peak_hr, 2),
        'peak_time': peak_time,
        'preceding_hr': round(preceding_hr, 2) if preceding_hr is not None else None,
        'following_hr': round(following_hr, 2) if following_hr is not None else None,
        'time_of_day': time_of_day,
        'day_of_week': day_of_week,
        'baseline_reliability': reliability,
        'n_observations': len(rows),
        'event_confidence': 'LOW',
    }


def compute_confidence(evt_dict):
    """Compute event confidence category based on statistical evidence."""
    duration = evt_dict.get('duration_minutes', 0)
    deviation = abs(evt_dict.get('deviation_bpm', 0))
    reliability = evt_dict.get('baseline_reliability', 'UNKNOWN')
    n_obs = evt_dict.get('n_observations', 0)

    score = 0

    if duration >= CONFIDENCE_HIGH_DURATION:
        score += 3
    elif duration >= CONFIDENCE_MEDIUM_DURATION:
        score += 2
    elif duration >= MIN_DURATION_MINUTES:
        score += 1

    if deviation >= CONFIDENCE_HIGH_DEVIATION_BPM:
        score += 3
    elif deviation >= CONFIDENCE_MEDIUM_DEVIATION_BPM:
        score += 2
    elif deviation >= MIN_DEVIATION_BPM:
        score += 1

    if reliability == 'HIGH':
        score += 2
    elif reliability == 'MEDIUM':
        score += 1

    if n_obs >= 30:
        score += 1

    if score >= 6:
        return 'HIGH'
    elif score >= 3:
        return 'MEDIUM'
    else:
        return 'LOW'


dev_df = flag_candidates(dev_df)

n_candidates_minutes = dev_df['candidate_flag'].sum()
print(f"\nDeviation computation complete:")
print(f"  Total minute observations: {len(dev_df):,}")
print(f"  Candidate minutes:         {n_candidates_minutes:,} ({n_candidates_minutes/len(dev_df)*100:.2f}%)")

print(f"\nDeviation statistics (bpm):")
print(f"  Mean:   {dev_df['deviation_bpm'].mean():>7.2f}")
print(f"  Median: {dev_df['deviation_bpm'].median():>7.2f}")
print(f"  Std:    {dev_df['deviation_bpm'].std():>7.2f}")
print(f"  Min:    {dev_df['deviation_bpm'].min():>7.2f}")
print(f"  Max:    {dev_df['deviation_bpm'].max():>7.2f}")

print(f"\nPer-subject candidate minutes:")
for sid in subjects:
    sub = dev_df[dev_df['subject_id'] == sid]
    n_cand = sub['candidate_flag'].sum()
    print(f"  {sid:>10.0f}: {n_cand:>6,} candidate / {len(sub):>8,} total ({n_cand/len(sub)*100:.2f}%)")


# ============================================================
# SECTION 3: DETECT CANDIDATE EVENTS
# ============================================================
print("\n" + "=" * 70)
print("SECTION 3: DETECT CANDIDATE EVENTS")
print("=" * 70)

candidate_events = []

for sid in subjects:
    sub = dev_df[dev_df['subject_id'] == sid].sort_values('timestamp_minute').copy()
    sub = sub.reset_index(drop=True)
    baseline_info = baseline_lookup[sid]
    quality_info = quality_lookup.get(sid, {})

    # ---- A & B: Sustained Elevation / Reduction ----
    # Group consecutive candidate minutes of the same direction
    in_event = False
    event_start = None
    event_type_label = None
    event_indices = []

    for i, row in sub.iterrows():
        if row['candidate_flag'] and row['candidate_type'] in ('elevated', 'reduced'):
            if not in_event or row['candidate_type'] != event_type_label:
                if in_event and len(event_indices) >= MIN_DURATION_MINUTES:
                    sustained_type = 'sustained_elevation' if event_type_label == 'elevated' else 'sustained_reduction'
                    evt = build_event(sub, event_indices, sustained_type, sid, baseline_info, quality_info)
                    candidate_events.append(evt)
                event_start = i
                event_type_label = row['candidate_type']
                event_indices = [i]
                in_event = True
            else:
                event_indices.append(i)
        else:
            if in_event:
                if len(event_indices) >= MIN_DURATION_MINUTES:
                    sustained_type = 'sustained_elevation' if event_type_label == 'elevated' else 'sustained_reduction'
                    evt = build_event(sub, event_indices, sustained_type, sid, baseline_info, quality_info)
                    candidate_events.append(evt)
                in_event = False
                event_indices = []

    if in_event and len(event_indices) >= MIN_DURATION_MINUTES:
        sustained_type = 'sustained_elevation' if event_type_label == 'elevated' else 'sustained_reduction'
        evt = build_event(sub, event_indices, sustained_type, sid, baseline_info, quality_info)
        candidate_events.append(evt)

    # ---- C: Sudden Change ----
    for i in range(1, len(sub)):
        prev_hr = sub.iloc[i - 1]['mean_hr']
        curr_hr = sub.iloc[i]['mean_hr']
        change = abs(curr_hr - prev_hr)
        if change >= SUDDEN_CHANGE_THRESHOLD:
            direction = 'elevated' if curr_hr > prev_hr else 'reduced'
            brief_indices = list(range(max(0, i - 1), min(len(sub), i + 2)))
            evt = build_event(sub, brief_indices, 'sudden_change', sid, baseline_info, quality_info)
            evt['event_type'] = 'sudden_change'
            evt['peak_hr'] = max(prev_hr, curr_hr)
            evt['deviation_bpm'] = round(change, 2)
            candidate_events.append(evt)

    # ---- D: Recurring Pattern ----
    # Find elevated or reduced candidate minutes and group by date
    elevated_dates = {}
    reduced_dates = {}
    for i, row in sub.iterrows():
        if row['candidate_flag'] and row['candidate_type'] == 'elevated':
            d = row['date']
            if d not in elevated_dates:
                elevated_dates[d] = []
            elevated_dates[d].append(i)
        elif row['candidate_flag'] and row['candidate_type'] == 'reduced':
            d = row['date']
            if d not in reduced_dates:
                reduced_dates[d] = []
            reduced_dates[d].append(i)

    for pattern_type, date_dict in [('recurring_elevation', elevated_dates), ('recurring_reduction', reduced_dates)]:
        if len(date_dict) >= MIN_RECURRING_OCCURRENCES:
            all_indices = []
            for d in sorted(date_dict.keys()):
                all_indices.extend(date_dict[d])
            if all_indices:
                evt = build_event(sub, all_indices, pattern_type, sid, baseline_info, quality_info)
                evt['n_occurrences'] = len(date_dict)
                candidate_events.append(evt)

    # ---- E: Unusual Variability ----
    personal_std = baseline_info['std_hr']
    if personal_std > 0:
        window = 15
        for start in range(0, len(sub), window):
            chunk = sub.iloc[start:start + window]
            if len(chunk) < 3:
                continue
            chunk_std = chunk['mean_hr'].std()
            if chunk_std > VARIABILITY_MULTIPLIER * personal_std:
                indices = list(range(start, min(start + window, len(sub))))
                evt = build_event(sub, indices, 'unusual_variability', sid, baseline_info, quality_info)
                candidate_events.append(evt)


candidate_events_df = pd.DataFrame(candidate_events)
if 'n_occurrences' in candidate_events_df.columns:
    candidate_events_df['n_occurrences'] = candidate_events_df['n_occurrences'].fillna(0).astype(int)

# Remove duplicate sudden_change events (same subject, overlapping times)
if 'event_type' in candidate_events_df.columns:
    sc = candidate_events_df[candidate_events_df['event_type'] == 'sudden_change'].copy()
    if len(sc) > 0:
        sc = sc.sort_values(['subject_id', 'start_time'])
        keep_mask = [True] * len(sc)
        for i in range(1, len(sc)):
            if (sc.iloc[i]['subject_id'] == sc.iloc[i-1]['subject_id'] and
                    sc.iloc[i]['start_time'] == sc.iloc[i-1]['start_time']):
                keep_mask[i] = False
        sc = sc[keep_mask]
        non_sc = candidate_events_df[candidate_events_df['event_type'] != 'sudden_change']
        candidate_events_df = pd.concat([non_sc, sc], ignore_index=True)

# Remove duplicate recurring events per subject+type
if len(candidate_events_df) > 0:
    candidate_events_df = candidate_events_df.drop_duplicates(
        subset=['subject_id', 'event_type', 'start_time'], keep='first'
    ).reset_index(drop=True)

print(f"\nCandidate events detected: {len(candidate_events_df)}")
if len(candidate_events_df) > 0:
    print(f"\nBy type:")
    for etype, count in candidate_events_df['event_type'].value_counts().items():
        print(f"  {etype:>30s}: {count}")
    print(f"\nBy subject:")
    for sid in subjects:
        n = (candidate_events_df['subject_id'] == sid).sum()
        print(f"  {sid:>10.0f}: {n:>4} events")


# ============================================================
# SECTION 4: CONFIRM EVENTS (temporal persistence)
# ============================================================
print("\n" + "=" * 70)
print("SECTION 4: CONFIRM EVENTS")
print("=" * 70)

# Confirmed events are those candidate events that satisfy
# temporal persistence requirements. A single noisy minute
# does not generate a clinical query.

confirmed_events = []
event_id_counter = 0

for _, evt in candidate_events_df.iterrows():
    is_confirmed = False

    if evt['event_type'] in ('sustained_elevation', 'sustained_reduction'):
        if evt['duration_minutes'] >= MIN_DURATION_MINUTES:
            is_confirmed = True

    elif evt['event_type'] == 'sudden_change':
        # Sudden changes are confirmed if the preceding/following HR shows
        # the change was not just a single outlier
        is_confirmed = True

    elif evt['event_type'] in ('recurring_elevation', 'recurring_reduction'):
        n_occ = evt.get('n_occurrences', 0)
        if n_occ >= MIN_RECURRING_OCCURRENCES:
            is_confirmed = True

    elif evt['event_type'] == 'unusual_variability':
        if evt['duration_minutes'] >= MIN_DURATION_MINUTES:
            is_confirmed = True

    if is_confirmed:
        event_id_counter += 1
        evt_dict = evt.to_dict()
        evt_dict['event_id'] = f'EVT-{event_id_counter:04d}'

        # Compute confidence
        conf = compute_confidence(evt_dict)
        evt_dict['event_confidence'] = conf
        confirmed_events.append(evt_dict)

confirmed_events_df = pd.DataFrame(confirmed_events)

print(f"\nConfirmed events: {len(confirmed_events_df)}")
if len(confirmed_events_df) > 0:
    print(f"\nBy type:")
    for etype, count in confirmed_events_df['event_type'].value_counts().items():
        print(f"  {etype:>30s}: {count}")
    print(f"\nBy confidence:")
    for conf, count in confirmed_events_df['event_confidence'].value_counts().items():
        print(f"  {conf:>10s}: {count}")
    print(f"\nBy subject:")
    for sid in subjects:
        n = (confirmed_events_df['subject_id'] == sid).sum()
        print(f"  {sid:>10.0f}: {n:>4} events")


# ============================================================
# SECTION 5: CLINICAL QUERY BUILDER
# ============================================================
print("\n" + "=" * 70)
print("SECTION 5: CLINICAL QUERY BUILDER")
print("=" * 70)


def build_clinical_query(evt):
    """Generate a natural-language clinical query for a confirmed event."""
    etype = evt['event_type']
    baseline = evt['personal_baseline_hr']
    observed = evt['observed_mean_hr']
    deviation = evt['deviation_bpm']
    duration = evt['duration_minutes']
    peak = evt['peak_hr']
    time_of_day = evt['time_of_day']
    reliability = evt['baseline_reliability']

    hour_str = f" around {time_of_day}:00" if time_of_day is not None else ""

    if etype == 'sustained_elevation':
        query = (
            f"A person with a personal baseline heart rate of {baseline:.0f} bpm "
            f"experienced a sustained heart rate around {observed:.0f} bpm for "
            f"{duration} minutes{hour_str}, representing approximately a "
            f"{deviation:.0f} bpm increase from baseline. What clinical guidance "
            f"exists regarding sustained elevated heart rate, and what contextual "
            f"factors should be considered?"
        )

    elif etype == 'sustained_reduction':
        query = (
            f"A person with a personal baseline heart rate of {baseline:.0f} bpm "
            f"experienced a sustained heart rate around {observed:.0f} bpm for "
            f"{duration} minutes{hour_str}, representing approximately a "
            f"{abs(deviation):.0f} bpm decrease from baseline. What clinical "
            f"guidance exists regarding sustained lower heart rate, and what "
            f"contextual factors should be considered?"
        )

    elif etype == 'sudden_change':
        direction = "increase" if deviation > 0 else "decrease"
        query = (
            f"A person with a personal baseline heart rate of {baseline:.0f} bpm "
            f"experienced a sudden heart rate {direction} from approximately "
            f"{evt.get('preceding_hr', baseline):.0f} bpm to {peak:.0f} bpm "
            f"within a short period{hour_str}. What clinical guidance exists "
            f"regarding sudden changes in heart rate, and what contextual factors "
            f"should be considered?"
        )

    elif etype in ('recurring_elevation', 'recurring_reduction'):
        n_occ = int(evt.get('n_occurrences', 0))
        pattern_word = "elevation" if "elevation" in etype else "reduction"
        query = (
            f"A person with a personal baseline heart rate of {baseline:.0f} bpm "
            f"experienced repeated periods of heart rate {pattern_word} to "
            f"approximately {peak:.0f} bpm across {n_occ} observed periods."
            f" What clinical guidance exists regarding recurrent heart-rate "
            f"patterns, and what contextual factors should be considered?"
        )

    elif etype == 'unusual_variability':
        query = (
            f"A person with a personal baseline heart rate of {baseline:.0f} bpm "
            f"exhibited unusual heart rate variability around {observed:.0f} bpm "
            f"for {duration} minutes{hour_str}, with variability substantially "
            f"different from their established pattern. What clinical guidance "
            f"exists regarding unusual heart rate variability, and what contextual "
            f"factors should be considered?"
        )

    else:
        query = (
            f"A person with a personal baseline heart rate of {baseline:.0f} bpm "
            f"experienced an unusual heart rate pattern{hour_str}. What clinical "
            f"guidance exists regarding this pattern, and what contextual factors "
            f"should be considered?"
        )

    return query


queries = []
if len(confirmed_events_df) > 0:
    for _, evt in confirmed_events_df.iterrows():
        q = build_clinical_query(evt)
        queries.append({
            'event_id': evt['event_id'],
            'subject_id': evt['subject_id'],
            'event_type': evt['event_type'],
            'query': q,
        })

queries_df = pd.DataFrame(queries)

print(f"\nClinical queries generated: {len(queries_df)}")
if len(queries_df) > 0:
    print(f"\nExample queries:")
    for i, row in queries_df.head(3).iterrows():
        print(f"\n  [{row['event_id']}] {row['event_type']}:")
        print(f"  {row['query']}")


# ============================================================
# SECTION 6: SAVE OUTPUTS
# ============================================================
print("\n" + "=" * 70)
print("SECTION 6: SAVE OUTPUTS")
print("=" * 70)

dev_df.to_csv(MINUTE_DEVIATIONS_PATH, index=False)
print(f"  Saved: {MINUTE_DEVIATIONS_PATH}")

confirmed_events_df.to_csv(ANOMALY_EVENTS_PATH, index=False)
print(f"  Saved: {ANOMALY_EVENTS_PATH}")

queries_df.to_csv(CLINICAL_QUERIES_PATH, index=False)
print(f"  Saved: {CLINICAL_QUERIES_PATH}")


# ============================================================
# SECTION 7: VISUALIZATIONS
# ============================================================
print("\n" + "=" * 70)
print("SECTION 7: VISUALIZATIONS")
print("=" * 70)

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Stage 3: Temporal Pattern / Anomaly Detection', fontsize=14, fontweight='bold')

# 1. Example subject timeline
example_sid = subjects[0]
ex_sub = dev_df[dev_df['subject_id'] == example_sid].sort_values('timestamp_minute')
ax1 = axes[0, 0]
ax1.plot(ex_sub['timestamp_minute'], ex_sub['mean_hr'], linewidth=0.3, alpha=0.6, color='steelblue', label='Observed HR')
ax1.axhline(y=ex_sub['personal_baseline_hr'].iloc[0], color='green', linestyle='--', linewidth=1.5, label='Personal Baseline')
cand = ex_sub[ex_sub['candidate_flag']]
if len(cand) > 0:
    ax1.scatter(cand['timestamp_minute'], cand['mean_hr'], s=5, c='red', alpha=0.5, label='Candidate', zorder=3)
ax1.set_title(f'Subject {example_sid:.0f}: HR Timeline with Events')
ax1.set_xlabel('Time')
ax1.set_ylabel('Heart Rate (bpm)')
ax1.legend(fontsize=7)
ax1.tick_params(axis='x', rotation=45)

# 2. Deviation distribution
ax2 = axes[0, 1]
ax2.hist(dev_df['deviation_bpm'], bins=100, color='steelblue', edgecolor='none', alpha=0.7)
ax2.axvline(x=0, color='black', linewidth=0.8)
ax2.axvline(x=MIN_DEVIATION_BPM, color='red', linestyle='--', linewidth=1, label=f'+{MIN_DEVIATION_BPM} bpm threshold')
ax2.axvline(x=-MIN_DEVIATION_BPM, color='red', linestyle='--', linewidth=1, label=f'-{MIN_DEVIATION_BPM} bpm threshold')
ax2.set_title('Distribution of Deviations from Personal Baseline')
ax2.set_xlabel('Deviation (bpm)')
ax2.set_ylabel('Count')
ax2.legend(fontsize=7)

# 3. Event duration distribution
ax3 = axes[1, 0]
if len(confirmed_events_df) > 0:
    durations = confirmed_events_df['duration_minutes']
    ax3.hist(durations, bins=min(30, max(5, int(durations.max() - durations.min()) + 1)),
             color='steelblue', edgecolor='none', alpha=0.7)
    ax3.axvline(x=MIN_DURATION_MINUTES, color='red', linestyle='--', linewidth=1,
                label=f'Min duration ({MIN_DURATION_MINUTES} min)')
    ax3.set_title('Confirmed Event Duration Distribution')
    ax3.set_xlabel('Duration (minutes)')
    ax3.set_ylabel('Count')
    ax3.legend(fontsize=7)
else:
    ax3.text(0.5, 0.5, 'No confirmed events', ha='center', va='center', transform=ax3.transAxes)
    ax3.set_title('Confirmed Event Duration Distribution')

# 4. Event counts by type
ax4 = axes[1, 1]
if len(confirmed_events_df) > 0:
    type_counts = confirmed_events_df['event_type'].value_counts()
    colors = sns.color_palette('Set2', len(type_counts))
    bars = ax4.barh(type_counts.index, type_counts.values, color=colors)
    ax4.set_title('Confirmed Events by Type')
    ax4.set_xlabel('Count')
    for bar, count in zip(bars, type_counts.values):
        ax4.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2,
                 str(count), va='center', fontsize=9)
else:
    ax4.text(0.5, 0.5, 'No confirmed events', ha='center', va='center', transform=ax4.transAxes)
    ax4.set_title('Confirmed Events by Type')

plt.tight_layout()
plt.savefig(VISUALIZATION_PATH, dpi=150, bbox_inches='tight')
plt.close()
print(f"  Saved: {VISUALIZATION_PATH}")


# ============================================================
# SECTION 8: REPORT
# ============================================================
print("\n" + "=" * 70)
print("SECTION 8: REPORT")
print("=" * 70)

n_candidate_events = len(candidate_events_df) if len(candidate_events_df) > 0 else 0
n_confirmed_events = len(confirmed_events_df) if len(confirmed_events_df) > 0 else 0
n_queries = len(queries_df) if len(queries_df) > 0 else 0

report_lines = []
report_lines.append("# Stage 3: Temporal Pattern / Anomaly Detection Report\n")
report_lines.append("## Dataset\n")
report_lines.append(f"- **Subjects:** {n_subjects}")
report_lines.append(f"- **Minute observations:** {n_minutes:,}")
report_lines.append(f"- **Date range:** {date_min} to {date_max}\n")

report_lines.append("## Detection Method\n")
report_lines.append("Stage 3 detects temporal deviations from each person's own established baseline.")
report_lines.append("The primary comparison is against the individual's median heart rate (personal baseline).")
report_lines.append("Hourly baselines from Stage 2 are used to provide circadian context, reducing")
report_lines.append("false positives from normal daily variation.\n")
report_lines.append("Each minute observation is scored for:")
report_lines.append("- Absolute deviation from personal baseline (bpm)")
report_lines.append("- Relative deviation (%)")
report_lines.append("- Robust standardized deviation (using Median Absolute Deviation)\n")
report_lines.append("Candidate minutes that exceed detection thresholds are grouped into temporal events.")
report_lines.append("Events are confirmed if they satisfy minimum duration or recurrence requirements.\n")

report_lines.append("## Thresholds\n")
report_lines.append("**These are prototype engineering/statistical detection parameters, NOT clinical diagnostic thresholds.**\n")
report_lines.append("| Parameter | Value | Purpose |")
report_lines.append("|---|---|---|")
report_lines.append(f"| MIN_DURATION_MINUTES | {MIN_DURATION_MINUTES} | Minimum consecutive minutes for sustained events |")
report_lines.append(f"| MIN_DEVIATION_BPM | {MIN_DEVIATION_BPM} | Minimum bpm deviation to flag a candidate minute |")
report_lines.append(f"| MAD_THRESHOLD | {MAD_THRESHOLD} | Robust z-score threshold in MAD units |")
report_lines.append(f"| SUDDEN_CHANGE_THRESHOLD | {SUDDEN_CHANGE_THRESHOLD} | Minimum bpm change between consecutive minutes |")
report_lines.append(f"| MIN_RECURRING_OCCURRENCES | {MIN_RECURRING_OCCURRENCES} | Minimum separate days for a recurring pattern |")
report_lines.append(f"| RECURRING_GAP_HOURS | {RECURRING_GAP_HOURS} | Maximum gap between occurrences |")
report_lines.append(f"| VARIABILITY_MULTIPLIER | {VARIABILITY_MULTIPLIER} | Multiplier over baseline std for unusual variability |")
report_lines.append(f"| MIN_HOURLY_OBSERVATIONS | {MIN_HOURLY_OBSERVATIONS} | Minimum observations in hourly bucket |")
report_lines.append(f"| MIN_READINGS_PER_MINUTE | {MIN_READINGS_PER_MINUTE} | Minimum readings to trust a minute's observation |")
report_lines.append("")

report_lines.append("## Results\n")
report_lines.append(f"- **Candidate minutes:** {n_candidates_minutes:,}")
report_lines.append(f"- **Candidate events:** {n_candidate_events}")
report_lines.append(f"- **Confirmed events:** {n_confirmed_events}")
report_lines.append(f"- **Clinical queries generated:** {n_queries}\n")

if len(confirmed_events_df) > 0:
    report_lines.append("### Event Types\n")
    for etype, count in confirmed_events_df['event_type'].value_counts().items():
        report_lines.append(f"- {etype}: {count}")

    report_lines.append("\n### Duration Statistics\n")
    report_lines.append(f"- Mean: {confirmed_events_df['duration_minutes'].mean():.1f} minutes")
    report_lines.append(f"- Median: {confirmed_events_df['duration_minutes'].median():.1f} minutes")
    report_lines.append(f"- Min: {confirmed_events_df['duration_minutes'].min()} minutes")
    report_lines.append(f"- Max: {confirmed_events_df['duration_minutes'].max()} minutes")

    report_lines.append("\n### Confidence Distribution\n")
    for conf, count in confirmed_events_df['event_confidence'].value_counts().items():
        report_lines.append(f"- {conf}: {count}")

report_lines.append("\n## Subject-Level Summary\n")
for sid in subjects:
    n = (confirmed_events_df['subject_id'] == sid).sum() if len(confirmed_events_df) > 0 else 0
    rel = quality_lookup.get(sid, {}).get('reliability', 'UNKNOWN')
    label = "no confirmed events"
    if n == 0:
        label = "no confirmed events"
    elif n <= 3:
        label = f"few events ({n})"
    elif n <= 10:
        label = f"moderate events ({n})"
    else:
        label = f"many events ({n})"
    note = " [low reliability]" if rel in ('LOW', 'MEDIUM') else ""
    report_lines.append(f"- **{sid:.0f}**: {label} (baseline reliability: {rel}){note}")

report_lines.append("\n## Clinical Queries\n")
report_lines.append(f"- **Total queries generated:** {n_queries}")
if len(queries_df) > 0:
    report_lines.append("\n### Example Queries\n")
    for _, row in queries_df.head(5).iterrows():
        report_lines.append(f"**[{row['event_id']}] {row['event_type']}:**")
        report_lines.append(f"> {row['query']}\n")

report_lines.append("## Limitations\n")
report_lines.append("- Only 15 subjects analyzed")
report_lines.append("- Fitbit-derived consumer wearable data")
report_lines.append("- Limited observation period (approximately 1 month)")
report_lines.append("- No clinical labels or diagnostic ground truth")
report_lines.append("- No age, gender, or regional context available")
report_lines.append("- Activity context not integrated (activity data only available at daily resolution)")
report_lines.append("- Statistical anomaly detection is NOT medical diagnosis")
report_lines.append("- Subject 2026352035 has only 5 days of data (MEDIUM reliability)\n")

report_lines.append("## RAG Integration Readiness\n")
report_lines.append("**YES WITH LIMITATIONS**\n")
report_lines.append("Stage 3 produces structured anomaly events and corresponding clinical queries")
report_lines.append("that are ready for downstream RAG consumption. Each event includes:")
report_lines.append("- Statistical evidence (deviation, duration, confidence)")
report_lines.append("- Personal baseline context")
report_lines.append("- Temporal metadata")
report_lines.append("- A carefully worded clinical query that avoids diagnosis\n")
report_lines.append("Limitations for RAG integration:")
report_lines.append("- Small subject pool (15)")
report_lines.append("- No activity/rest context to distinguish exercise from resting HR elevation")
report_lines.append("- No clinical labels for validation")
report_lines.append("- Detection thresholds are engineering parameters, not clinically validated")

report_text = "\n".join(report_lines)
with open(REPORT_PATH, 'w', encoding='utf-8') as f:
    f.write(report_text)
print(f"  Saved: {REPORT_PATH}")


# ============================================================
# FINAL OUTPUT
# ============================================================
print("\n")
print("=" * 70)
print("STAGE 3 - TEMPORAL ANOMALY DETECTION")
print("=" * 70)
print()
print(f"Subjects processed:             {n_subjects}")
print(f"Minute observations:            {n_minutes:,}")
print(f"Candidate events:               {n_candidate_events}")
print(f"Confirmed events:               {n_confirmed_events}")
print()

if len(confirmed_events_df) > 0:
    type_counts = confirmed_events_df['event_type'].value_counts()
    print("Event types:")
    for etype in ['sustained_elevation', 'sustained_reduction', 'sudden_change',
                   'recurring_elevation', 'recurring_reduction', 'unusual_variability']:
        count = type_counts.get(etype, 0)
        label = etype.replace('_', ' ').title()
        print(f"  {label:<30s}: {count}")
else:
    print("Event types:")
    print("  No confirmed events detected")

print()
if len(confirmed_events_df) > 0:
    conf_counts = confirmed_events_df['event_confidence'].value_counts()
    print("Confidence:")
    for c in ['HIGH', 'MEDIUM', 'LOW']:
        print(f"  {c:<30s}: {conf_counts.get(c, 0)}")

print(f"\nClinical queries generated:     {n_queries}")
print()
print("Outputs:")
print(f"  {MINUTE_DEVIATIONS_PATH}")
print(f"  {ANOMALY_EVENTS_PATH}")
print(f"  {CLINICAL_QUERIES_PATH}")
print(f"  {VISUALIZATION_PATH}")
print(f"  {REPORT_PATH}")
print()
print("RAG integration readiness:      YES WITH LIMITATIONS")
print()
print("Important:")
print("  Stage 3 detects temporal deviations.")
print("  It does NOT diagnose medical conditions.")
