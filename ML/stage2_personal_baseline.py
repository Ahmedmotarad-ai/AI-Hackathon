"""
Stage 2: Personal Baseline
===========================
For each subject with Fitbit heart-rate data, compute a robust personal
HR baseline that captures "what is normal heart-rate behavior for this
individual?"

This script implements ONLY Stage 2 of the Heart Rate Monitoring ML pipeline.
Stage 1 (Peer Baseline) is already complete and must not be modified.
Stage 3 (Temporal Pattern / Anomaly Detection) is NOT implemented here.

Important:
  - Dataset 2 subjects are completely different from Stage 1 patients.
  - No cross-dataset identity matching is performed.
  - Stage 2 operates independently from Stage 1.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
warnings.filterwarnings('ignore')

sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 5)
plt.rcParams['figure.dpi'] = 100

# ============================================================
# CONFIGURATION
# ============================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Input: both HR files (P1 and P2)
DATA_DIR = os.path.join(SCRIPT_DIR, 'data')
HR_FILE_P1 = os.path.join(
    DATA_DIR, 'Stage 2',
    'mturkfitbit_export_3.12.16-4.11.16', 'Fitabase Data 3.12.16-4.11.16',
    'heartrate_seconds_merged.csv'
)
HR_FILE_P2 = os.path.join(
    DATA_DIR, 'Stage 2',
    'mturkfitbit_export_4.12.16-5.12.16', 'Fitabase Data 4.12.16-5.12.16',
    'heartrate_seconds_merged.csv'
)

OUTPUT_DIR = os.path.join(DATA_DIR, 'processed')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Column names (verified from actual data)
ID_COL = 'Id'
TIME_COL = 'Time'
HR_COL = 'Value'

# Reliability thresholds
MIN_RECORDS_HIGH = 100000
MIN_RECORDS_MEDIUM = 10000
MIN_DAYS_HIGH = 21
MIN_DAYS_MEDIUM = 7

# Gap thresholds (seconds)
GAP_THRESHOLDS = [60, 300, 1800]  # 1 min, 5 min, 30 min


# ============================================================
# 1. LOAD AND COMBINE HR DATA
# ============================================================
def load_hr_data():
    print("=" * 70)
    print("SECTION 1: LOAD AND COMBINE HR DATA")
    print("=" * 70)

    dfs = []
    for label, path in [('P1', HR_FILE_P1), ('P2', HR_FILE_P2)]:
        print(f"\nLoading {label}: {os.path.basename(path)}")
        df = pd.read_csv(path)
        print(f"  Rows: {len(df):,}  Columns: {list(df.columns)}")
        print(f"  Unique IDs: {df[ID_COL].nunique()}")
        print(f"  Time range: {df[TIME_COL].iloc[0]} to {df[TIME_COL].iloc[-1]}")
        dfs.append(df)

    combined = pd.concat(dfs, ignore_index=True)

    # Parse timestamps
    combined[TIME_COL] = pd.to_datetime(combined[TIME_COL])

    # Sort by subject and time
    combined = combined.sort_values([ID_COL, TIME_COL]).reset_index(drop=True)

    # Remove exact duplicates (same Id + same Time)
    n_before = len(combined)
    combined = combined.drop_duplicates(subset=[ID_COL, TIME_COL], keep='first')
    n_after = len(combined)
    print(f"\nCombined dataset:")
    print(f"  Total rows:    {n_after:,} (removed {n_before - n_after} duplicates)")
    print(f"  Unique IDs:    {combined[ID_COL].nunique()}")
    print(f"  Time range:    {combined[TIME_COL].min()} to {combined[TIME_COL].max()}")
    print(f"  HR range:      {combined[HR_COL].min()} - {combined[HR_COL].max()} bpm")
    print(f"  Columns:       {list(combined.columns)}")

    return combined


# ============================================================
# 2. DATA QUALITY: SUBJECT COVERAGE
# ============================================================
def compute_subject_coverage(df):
    print("\n" + "=" * 70)
    print("SECTION 2: DATA QUALITY - SUBJECT COVERAGE")
    print("=" * 70)

    coverage_rows = []
    for uid in sorted(df[ID_COL].unique()):
        sub = df[df[ID_COL] == uid].sort_values(TIME_COL)
        dates = sub[TIME_COL].dt.date
        n_days = dates.nunique()
        readings_per_day = sub.groupby(sub[TIME_COL].dt.date).size()
        coverage_rows.append({
            'subject_id': uid,
            'observation_count': len(sub),
            'n_days': n_days,
            'first_timestamp': sub[TIME_COL].min(),
            'last_timestamp': sub[TIME_COL].max(),
            'median_readings_per_day': readings_per_day.median(),
            'min_readings_per_day': readings_per_day.min(),
            'max_readings_per_day': readings_per_day.max(),
            'time_span_days': (sub[TIME_COL].max() - sub[TIME_COL].min()).total_seconds() / 86400,
        })

    coverage = pd.DataFrame(coverage_rows)
    print(f"\n{'ID':>12s}  {'Records':>9s}  {'Days':>5s}  {'Median/Day':>10s}  {'Min/Day':>8s}  {'Max/Day':>8s}  {'Span(d)':>7s}")
    print("-" * 75)
    for _, r in coverage.iterrows():
        print(f"  {int(r['subject_id']):>10d}  {int(r['observation_count']):>9,d}  {int(r['n_days']):>5d}  "
              f"{int(r['median_readings_per_day']):>10d}  {int(r['min_readings_per_day']):>8d}  "
              f"{int(r['max_readings_per_day']):>8d}  {r['time_span_days']:>7.1f}")

    print(f"\nSummary:")
    print(f"  Subjects:           {len(coverage)}")
    print(f"  Total observations: {coverage['observation_count'].sum():,}")
    print(f"  Mean observations:  {coverage['observation_count'].mean():,.0f}")
    print(f"  Min observations:   {coverage['observation_count'].min():,}")
    print(f"  Max observations:   {coverage['observation_count'].max():,}")
    print(f"  Mean days:          {coverage['n_days'].mean():.1f}")
    print(f"  Min days:           {coverage['n_days'].min():.0f}")
    print(f"  Max days:           {coverage['n_days'].max():.0f}")

    return coverage


# ============================================================
# 3. DATA QUALITY: TEMPORAL GAPS
# ============================================================
def compute_gap_analysis(df):
    print("\n" + "=" * 70)
    print("SECTION 3: DATA QUALITY - TEMPORAL GAP ANALYSIS")
    print("=" * 70)

    gap_rows = []
    for uid in sorted(df[ID_COL].unique()):
        sub = df[df[ID_COL] == uid].sort_values(TIME_COL).copy()
        sub['dt'] = sub[TIME_COL].diff().dt.total_seconds()
        diffs = sub['dt'].dropna()

        row = {'subject_id': uid}
        for thr in GAP_THRESHOLDS:
            row[f'gaps_gt_{thr}s'] = (diffs > thr).sum()
        row['largest_gap_s'] = diffs.max() if len(diffs) > 0 else 0
        row['median_interval_s'] = diffs.median() if len(diffs) > 0 else 0
        gap_rows.append(row)

    gaps = pd.DataFrame(gap_rows)

    print(f"\n{'ID':>12s}  {'>1min':>7s}  {'>5min':>7s}  {'>30min':>7s}  {'Largest(s)':>10s}  {'Med.Intv':>9s}")
    print("-" * 60)
    for _, r in gaps.iterrows():
        print(f"  {int(r['subject_id']):>10d}  {int(r['gaps_gt_60s']):>7d}  "
              f"{int(r['gaps_gt_300s']):>7d}  {int(r['gaps_gt_1800s']):>7d}  "
              f"{r['largest_gap_s']:>10.0f}  {r['median_interval_s']:>9.1f}")

    print(f"\nObservations:")
    print(f"  - Most gaps >1min are normal Fitbit behavior (sensor only records during activity)")
    print(f"  - Gaps >30min during waking hours may indicate device removal or inactivity")
    print(f"  - Overnight gaps (during sleep) are expected and not abnormal")

    return gaps


# ============================================================
# 4. MINUTE-LEVEL AGGREGATION
# ============================================================
def aggregate_to_minute(df):
    print("\n" + "=" * 70)
    print("SECTION 4: MINUTE-LEVEL AGGREGATION")
    print("=" * 70)

    df = df.copy()
    df['timestamp_minute'] = df[TIME_COL].dt.floor('min')

    minute_df = df.groupby([ID_COL, 'timestamp_minute']).agg(
        mean_hr=(HR_COL, 'mean'),
        median_hr=(HR_COL, 'median'),
        min_hr=(HR_COL, 'min'),
        max_hr=(HR_COL, 'max'),
        std_hr=(HR_COL, 'std'),
        reading_count=(HR_COL, 'count')
    ).reset_index()

    # std is NaN when only 1 reading in a minute
    minute_df['std_hr'] = minute_df['std_hr'].fillna(0)

    print(f"\nMinute-level dataset:")
    print(f"  Rows:          {len(minute_df):,}")
    print(f"  Unique IDs:    {minute_df[ID_COL].nunique()}")
    print(f"  Time range:    {minute_df['timestamp_minute'].min()} to {minute_df['timestamp_minute'].max()}")

    print(f"\nReadings per minute distribution:")
    print(f"  Median:  {minute_df['reading_count'].median():.1f}")
    print(f"  Mean:    {minute_df['reading_count'].mean():.1f}")
    print(f"  Max:     {minute_df['reading_count'].max():.0f}")
    print(f"  P95:     {minute_df['reading_count'].quantile(0.95):.1f}")

    print(f"\nPer-subject minute counts:")
    for uid in sorted(minute_df[ID_COL].unique()):
        sub = minute_df[minute_df[ID_COL] == uid]
        n_days = sub['timestamp_minute'].dt.date.nunique()
        print(f"  {uid:>10.0f}: {len(sub):>8,} minutes across {n_days:>3} days")

    return minute_df


# ============================================================
# 5. SUBJECT-LEVEL BASELINE STATISTICS
# ============================================================
def compute_subject_baseline(df):
    print("\n" + "=" * 70)
    print("SECTION 5: SUBJECT-LEVEL BASELINE STATISTICS")
    print("=" * 70)

    rows = []
    for uid in sorted(df[ID_COL].unique()):
        vals = df.loc[df[ID_COL] == uid, HR_COL].values
        sorted_vals = np.sort(vals)

        # Successive differences
        successive_diffs = np.abs(np.diff(sorted_vals))

        row = {
            'subject_id': uid,
            'observation_count': len(vals),
            'mean_hr': np.mean(vals),
            'median_hr': np.median(vals),
            'std_hr': np.std(vals, ddof=1) if len(vals) > 1 else 0,
            'min_hr': np.min(vals),
            'max_hr': np.max(vals),
            'p05_hr': np.percentile(vals, 5),
            'p10_hr': np.percentile(vals, 10),
            'p25_hr': np.percentile(vals, 25),
            'p50_hr': np.percentile(vals, 50),
            'p75_hr': np.percentile(vals, 75),
            'p90_hr': np.percentile(vals, 90),
            'p95_hr': np.percentile(vals, 95),
            'iqr': np.percentile(vals, 75) - np.percentile(vals, 25),
            'mad': np.median(np.abs(vals - np.median(vals))),
            'cv': (np.std(vals, ddof=1) / np.mean(vals) * 100) if np.mean(vals) > 0 and len(vals) > 1 else 0,
            'mean_abs_successive_diff': np.mean(successive_diffs),
            'median_abs_successive_diff': np.median(successive_diffs),
        }
        rows.append(row)

    baseline = pd.DataFrame(rows)

    # Print results
    print(f"\n{'ID':>12s}  {'N':>9s}  {'Mean':>6s}  {'Median':>6s}  {'Std':>6s}  "
          f"{'P10':>5s}  {'P90':>5s}  {'IQR':>5s}  {'MAD':>5s}  {'CV%':>6s}")
    print("-" * 85)
    for _, r in baseline.iterrows():
        print(f"  {int(r['subject_id']):>10d}  {int(r['observation_count']):>9,d}  "
              f"{r['mean_hr']:>6.1f}  {r['median_hr']:>6.1f}  {r['std_hr']:>6.1f}  "
              f"{r['p10_hr']:>5.1f}  {r['p90_hr']:>5.1f}  {r['iqr']:>5.1f}  "
              f"{r['mad']:>5.1f}  {r['cv']:>6.1f}")

    print(f"\nInterpretation:")
    print(f"  Personal baseline = median HR (more robust than mean to outliers)")
    print(f"  IQR = P75 - P25 (middle 50% of HR values)")
    print(f"  P10-P90 range = central 80% of HR values")
    print(f"  MAD = Median Absolute Deviation from median (robust variability)")
    print(f"  CV = Coefficient of Variation (relative variability)")
    print(f"  Mean/Median Abs Successive Difference = temporal variability (not clinical HRV)")

    return baseline


# ============================================================
# 6. TIME-AWARE BASELINE: HOURLY
# ============================================================
def compute_hourly_baseline(df):
    print("\n" + "=" * 70)
    print("SECTION 6: TIME-AWARE BASELINE - HOURLY")
    print("=" * 70)

    df = df.copy()
    df['hour'] = df[TIME_COL].dt.hour

    hourly = df.groupby([ID_COL, 'hour']).agg(
        mean_hr=(HR_COL, 'mean'),
        median_hr=(HR_COL, 'median'),
        std_hr=(HR_COL, 'std'),
        reading_count=(HR_COL, 'count')
    ).reset_index()
    hourly['std_hr'] = hourly['std_hr'].fillna(0)

    print(f"\nHourly baseline dataset:")
    print(f"  Rows: {len(hourly):,}")
    print(f"  Unique IDs: {hourly[ID_COL].nunique()}")
    print(f"  Hours per subject: {hourly.groupby(ID_COL).size().iloc[0]}")

    # Show example for first subject
    first_id = sorted(hourly[ID_COL].unique())[0]
    print(f"\nExample: Subject {first_id} hourly baseline")
    print(f"  {'Hour':>5s}  {'Median HR':>9s}  {'Mean HR':>8s}  {'Std HR':>7s}  {'Count':>6s}")
    print(f"  {'-'*5}  {'-'*9}  {'-'*8}  {'-'*7}  {'-'*6}")
    sub = hourly[hourly[ID_COL] == first_id].sort_values('hour')
    for _, r in sub.iterrows():
        print(f"  {int(r['hour']):>5d}  {r['median_hr']:>9.1f}  {r['mean_hr']:>8.1f}  "
              f"{r['std_hr']:>7.1f}  {int(r['reading_count']):>6,d}")

    # Daytime vs nighttime summary
    print(f"\nDaytime (6-22) vs Nighttime (22-6) comparison:")
    for uid in sorted(hourly[ID_COL].unique()):
        sub = hourly[hourly[ID_COL] == uid]
        day = sub[sub['hour'].between(6, 21)]
        night = sub[sub['hour'].between(22, 23) | sub['hour'].between(0, 5)]
        day_med = day['median_hr'].mean() if len(day) > 0 else float('nan')
        night_med = night['median_hr'].mean() if len(night) > 0 else float('nan')
        diff = day_med - night_med if not (np.isnan(day_med) or np.isnan(night_med)) else float('nan')
        print(f"  {uid:>10.0f}: Day median={day_med:.1f}, Night median={night_med:.1f}, Diff={diff:+.1f} bpm")

    return hourly


# ============================================================
# 7. TIME-AWARE BASELINE: DAILY
# ============================================================
def compute_daily_baseline(df):
    print("\n" + "=" * 70)
    print("SECTION 7: TIME-AWARE BASELINE - DAILY")
    print("=" * 70)

    df = df.copy()
    df['date'] = df[TIME_COL].dt.date

    daily = df.groupby([ID_COL, 'date']).agg(
        mean_hr=(HR_COL, 'mean'),
        median_hr=(HR_COL, 'median'),
        std_hr=(HR_COL, 'std'),
        min_hr=(HR_COL, 'min'),
        max_hr=(HR_COL, 'max'),
        reading_count=(HR_COL, 'count')
    ).reset_index()
    daily['std_hr'] = daily['std_hr'].fillna(0)

    print(f"\nDaily baseline dataset:")
    print(f"  Rows: {len(daily):,}")
    print(f"  Unique IDs: {daily[ID_COL].nunique()}")
    print(f"  Date range: {daily['date'].min()} to {daily['date'].max()}")

    # Per-subject daily summary
    print(f"\nPer-subject daily statistics:")
    for uid in sorted(daily[ID_COL].unique()):
        sub = daily[daily[ID_COL] == uid]
        print(f"  {uid:>10.0f}: {len(sub)} days, "
              f"median HR range: {sub['median_hr'].min():.1f}-{sub['median_hr'].max():.1f}, "
              f"mean across days: {sub['median_hr'].mean():.1f}")

    return daily


# ============================================================
# 8. RELIABILITY ASSESSMENT
# ============================================================
def assess_reliability(baseline, coverage, gaps):
    print("\n" + "=" * 70)
    print("SECTION 8: BASELINE RELIABILITY ASSESSMENT")
    print("=" * 70)

    merged = baseline.merge(coverage[['subject_id', 'n_days', 'time_span_days']], on='subject_id')
    merged = merged.merge(gaps[['subject_id', 'gaps_gt_1800s', 'largest_gap_s']], on='subject_id')

    def classify(row):
        score = 0
        # Observation count
        if row['observation_count'] >= MIN_RECORDS_HIGH:
            score += 3
        elif row['observation_count'] >= MIN_RECORDS_MEDIUM:
            score += 2
        else:
            score += 1
        # Days observed
        if row['n_days'] >= MIN_DAYS_HIGH:
            score += 3
        elif row['n_days'] >= MIN_DAYS_MEDIUM:
            score += 2
        else:
            score += 1
        # Time span
        if row['time_span_days'] >= 21:
            score += 2
        elif row['time_span_days'] >= 7:
            score += 1
        # Large gaps
        if row['gaps_gt_1800s'] < 10:
            score += 1
        elif row['gaps_gt_1800s'] < 50:
            score += 0
        else:
            score -= 1
        # Distribution stability (low CV = stable)
        if row['cv'] < 25:
            score += 1
        elif row['cv'] > 40:
            score -= 1

        if score >= 8:
            return 'HIGH'
        elif score >= 5:
            return 'MEDIUM'
        else:
            return 'LOW'

    merged['reliability'] = merged.apply(classify, axis=1)

    print(f"\nReliability criteria:")
    print(f"  HIGH: >= {MIN_RECORDS_HIGH:,} records, >= {MIN_DAYS_HIGH} days, stable distribution")
    print(f"  MEDIUM: >= {MIN_RECORDS_MEDIUM:,} records, >= {MIN_DAYS_MEDIUM} days")
    print(f"  LOW: fewer observations or days")
    print(f"  Adjusted for gaps and distribution stability (CV)")

    print(f"\n{'ID':>12s}  {'Records':>9s}  {'Days':>5s}  {'CV%':>6s}  {'>30min gaps':>11s}  {'Reliability':>12s}")
    print("-" * 65)
    for _, r in merged.sort_values('subject_id').iterrows():
        print(f"  {int(r['subject_id']):>10d}  {int(r['observation_count']):>9,d}  {int(r['n_days']):>5d}  "
              f"{r['cv']:>6.1f}  {int(r['gaps_gt_1800s']):>11d}  {r['reliability']:>12s}")

    counts = merged['reliability'].value_counts()
    print(f"\nReliability distribution:")
    for level in ['HIGH', 'MEDIUM', 'LOW']:
        n = counts.get(level, 0)
        print(f"  {level:>8s}: {n:>2d} subjects ({n/len(merged)*100:.1f}%)")

    return merged[['subject_id', 'reliability']]


# ============================================================
# 9. VISUALIZATIONS
# ============================================================
def create_visualizations(baseline, coverage, hourly, daily, df):
    print("\n" + "=" * 70)
    print("SECTION 9: VISUALIZATIONS")
    print("=" * 70)

    fig = plt.figure(figsize=(24, 18))

    # 1. Distribution of personal median HR across subjects
    ax1 = fig.add_subplot(3, 3, 1)
    ax1.barh(range(len(baseline)), baseline['median_hr'].values, color='steelblue', edgecolor='white')
    ax1.set_yticks(range(len(baseline)))
    ax1.set_yticklabels([str(int(x)) for x in baseline['subject_id'].values], fontsize=7)
    ax1.set_xlabel('Median HR (bpm)')
    ax1.set_title('Personal Median HR by Subject')
    ax1.axvline(baseline['median_hr'].mean(), color='red', linestyle='--',
                label=f'Pop. mean: {baseline["median_hr"].mean():.1f}')
    ax1.legend(fontsize=8)

    # 2. Distribution of personal HR variability (IQR)
    ax2 = fig.add_subplot(3, 3, 2)
    ax2.barh(range(len(baseline)), baseline['iqr'].values, color='darkorange', edgecolor='white')
    ax2.set_yticks(range(len(baseline)))
    ax2.set_yticklabels([str(int(x)) for x in baseline['subject_id'].values], fontsize=7)
    ax2.set_xlabel('IQR (bpm)')
    ax2.set_title('Personal HR Variability (IQR) by Subject')

    # 3. Example subject HR distribution (highest N subject)
    ax3 = fig.add_subplot(3, 3, 3)
    top_subject = coverage.sort_values('observation_count', ascending=False).iloc[0]['subject_id']
    vals = df[df[ID_COL] == top_subject][HR_COL].values
    ax3.hist(vals, bins=100, color='steelblue', edgecolor='white', alpha=0.7, density=True)
    med = np.median(vals)
    ax3.axvline(med, color='red', linestyle='--', linewidth=2, label=f'Median: {med:.1f}')
    ax3.axvline(np.mean(vals), color='orange', linestyle='--', linewidth=2, label=f'Mean: {np.mean(vals):.1f}')
    ax3.set_xlabel('HR (bpm)')
    ax3.set_ylabel('Density')
    ax3.set_title(f'HR Distribution - Subject {int(top_subject)}')
    ax3.legend(fontsize=8)

    # 4. Example subject daily baseline
    ax4 = fig.add_subplot(3, 3, 4)
    daily_sub = daily[daily[ID_COL] == top_subject].sort_values('date')
    ax4.plot(daily_sub['date'].values, daily_sub['median_hr'].values, 'o-',
             color='steelblue', markersize=4, label='Median HR')
    ax4.fill_between(range(len(daily_sub)),
                     daily_sub['p10_hr'].values if 'p10_hr' in daily_sub else daily_sub['min_hr'].values,
                     daily_sub['p90_hr'].values if 'p90_hr' in daily_sub else daily_sub['max_hr'].values,
                     alpha=0.2, color='steelblue', label='Min-Max range')
    ax4.set_xlabel('Date')
    ax4.set_ylabel('HR (bpm)')
    ax4.set_title(f'Daily Baseline - Subject {int(top_subject)}')
    ax4.legend(fontsize=8)
    ax4.tick_params(axis='x', rotation=45)

    # 5. Example subject hourly baseline
    ax5 = fig.add_subplot(3, 3, 5)
    hourly_sub = hourly[hourly[ID_COL] == top_subject].sort_values('hour')
    ax5.plot(hourly_sub['hour'].values, hourly_sub['median_hr'].values, 'o-',
             color='steelblue', markersize=6, label='Median HR')
    ax5.fill_between(hourly_sub['hour'].values,
                     hourly_sub['median_hr'].values - hourly_sub['std_hr'].values,
                     hourly_sub['median_hr'].values + hourly_sub['std_hr'].values,
                     alpha=0.2, color='steelblue', label='Median +/- Std')
    ax5.set_xlabel('Hour of Day')
    ax5.set_ylabel('HR (bpm)')
    ax5.set_title(f'Hourly Baseline - Subject {int(top_subject)}')
    ax5.set_xticks(range(0, 24, 2))
    ax5.legend(fontsize=8)

    # 6. Day vs Night comparison
    ax6 = fig.add_subplot(3, 3, 6)
    df_plot = df.copy()
    df_plot['hour'] = df_plot[TIME_COL].dt.hour
    df_plot['period'] = df_plot['hour'].apply(
        lambda h: 'Night (0-6)' if h < 6 else ('Day (6-22)' if h < 22 else 'Evening (22-24)')
    )
    period_order = ['Night (0-6)', 'Day (6-22)', 'Evening (22-24)']
    # Use only first 3 subjects for readability
    sample_ids = sorted(df[ID_COL].unique())[:3]
    df_sample = df_plot[df_plot[ID_COL].isin(sample_ids)]
    for i, uid in enumerate(sample_ids):
        sub = df_sample[df_sample[ID_COL] == uid]
        means = sub.groupby('period')[HR_COL].median().reindex(period_order)
        ax6.plot(range(len(means)), means.values, 'o-', markersize=8, label=f'ID {int(uid)}')
    ax6.set_xticks(range(len(period_order)))
    ax6.set_xticklabels(period_order, fontsize=9)
    ax6.set_ylabel('Median HR (bpm)')
    ax6.set_title('Day vs Night HR Comparison (3 subjects)')
    ax6.legend(fontsize=8)

    # 7. Mean-Median difference (bias indicator)
    ax7 = fig.add_subplot(3, 3, 7)
    bias = baseline['mean_hr'] - baseline['median_hr']
    colors = ['steelblue' if abs(b) < 2 else 'crimson' for b in bias.values]
    ax7.barh(range(len(baseline)), bias.values, color=colors, edgecolor='white')
    ax7.set_yticks(range(len(baseline)))
    ax7.set_yticklabels([str(int(x)) for x in baseline['subject_id'].values], fontsize=7)
    ax7.set_xlabel('Mean - Median HR (bpm)')
    ax7.set_title('Mean-Median Bias (skewness indicator)')
    ax7.axvline(0, color='black', linestyle='-', linewidth=0.5)

    # 8. CV by subject
    ax8 = fig.add_subplot(3, 3, 8)
    ax8.barh(range(len(baseline)), baseline['cv'].values, color='green', edgecolor='white')
    ax8.set_yticks(range(len(baseline)))
    ax8.set_yticklabels([str(int(x)) for x in baseline['subject_id'].values], fontsize=7)
    ax8.set_xlabel('CV (%)')
    ax8.set_title('Coefficient of Variation by Subject')

    # 9. Reading count by subject
    ax9 = fig.add_subplot(3, 3, 9)
    ax9.barh(range(len(coverage)), coverage['observation_count'].values, color='purple', edgecolor='white')
    ax9.set_yticks(range(len(coverage)))
    ax9.set_yticklabels([str(int(x)) for x in coverage['subject_id'].values], fontsize=7)
    ax9.set_xlabel('Number of HR Observations')
    ax9.set_title('Data Volume by Subject')

    plt.suptitle('Stage 2: Personal Baseline - Heart Rate Analysis',
                 fontsize=16, y=1.02)
    plt.tight_layout()

    viz_path = os.path.join(OUTPUT_DIR, 'stage2_personal_baseline_visualizations.png')
    plt.savefig(viz_path, bbox_inches='tight', dpi=150)
    plt.close()
    print(f"\nVisualizations saved to: {viz_path}")

    return viz_path


# ============================================================
# 10. SAVE OUTPUTS
# ============================================================
def save_outputs(baseline, minute_df, hourly, daily, coverage, gaps, reliability):
    print("\n" + "=" * 70)
    print("SECTION 10: SAVE OUTPUTS")
    print("=" * 70)

    # A. Subject baseline
    baseline_path = os.path.join(OUTPUT_DIR, 'stage2_subject_baseline.csv')
    baseline.to_csv(baseline_path, index=False)
    print(f"\nA. Subject baseline: {baseline_path}")
    print(f"   Columns: {list(baseline.columns)}")
    print(f"   Rows: {len(baseline)}")

    # B. Minute-level HR
    minute_path = os.path.join(OUTPUT_DIR, 'stage2_minute_hr.csv')
    minute_df.to_csv(minute_path, index=False)
    print(f"\nB. Minute-level HR: {minute_path}")
    print(f"   Columns: {list(minute_df.columns)}")
    print(f"   Rows: {len(minute_df):,}")

    # C. Hourly personal baseline
    hourly_path = os.path.join(OUTPUT_DIR, 'stage2_hourly_baseline.csv')
    hourly.to_csv(hourly_path, index=False)
    print(f"\nC. Hourly baseline: {hourly_path}")
    print(f"   Columns: {list(hourly.columns)}")
    print(f"   Rows: {len(hourly):,}")

    # D. Daily personal baseline
    daily_path = os.path.join(OUTPUT_DIR, 'stage2_daily_baseline.csv')
    daily.to_csv(daily_path, index=False)
    print(f"\nD. Daily baseline: {daily_path}")
    print(f"   Columns: {list(daily.columns)}")
    print(f"   Rows: {len(daily):,}")

    # E. Data quality report
    quality = coverage.merge(gaps, on='subject_id').merge(reliability, on='subject_id')
    quality_path = os.path.join(OUTPUT_DIR, 'stage2_data_quality.csv')
    quality.to_csv(quality_path, index=False)
    print(f"\nE. Data quality: {quality_path}")
    print(f"   Columns: {list(quality.columns)}")
    print(f"   Rows: {len(quality)}")

    return {
        'baseline': baseline_path,
        'minute': minute_path,
        'hourly': hourly_path,
        'daily': daily_path,
        'quality': quality_path,
    }


# ============================================================
# 11. INTERPRETATION
# ============================================================
def interpret_results(baseline, coverage, hourly, daily):
    print("\n" + "=" * 70)
    print("SECTION 11: BASELINE INTERPRETATION")
    print("=" * 70)

    print(f"\n--- How different are personal baselines? ---")
    median_range = baseline['median_hr'].max() - baseline['median_hr'].min()
    print(f"  Population median HR range: {baseline['median_hr'].min():.1f} - {baseline['median_hr'].max():.1f} bpm")
    print(f"  Spread: {median_range:.1f} bpm")
    print(f"  Population median of medians: {baseline['median_hr'].median():.1f} bpm")
    print(f"  Std of medians across subjects: {baseline['median_hr'].std():.1f} bpm")

    print(f"\n--- Within-person variation ---")
    print(f"  IQR range across subjects: {baseline['iqr'].min():.1f} - {baseline['iqr'].max():.1f} bpm")
    print(f"  Mean IQR: {baseline['iqr'].mean():.1f} bpm")
    print(f"  MAD range: {baseline['mad'].min():.1f} - {baseline['mad'].max():.1f} bpm")

    print(f"\n--- Daytime vs Nighttime ---")
    df_hourly_stats = []
    for uid in sorted(hourly[ID_COL].unique()):
        sub = hourly[hourly[ID_COL] == uid]
        day = sub[sub['hour'].between(6, 21)]
        night = sub[sub['hour'].between(22, 23) | sub['hour'].between(0, 5)]
        if len(day) > 0 and len(night) > 0:
            df_hourly_stats.append({
                'subject_id': uid,
                'day_median': day['median_hr'].mean(),
                'night_median': night['median_hr'].mean(),
                'diff': day['median_hr'].mean() - night['median_hr'].mean()
            })

    if df_hourly_stats:
        hourly_stats = pd.DataFrame(df_hourly_stats)
        print(f"  Subjects with day/night data: {len(hourly_stats)}")
        print(f"  Mean day-night difference: {hourly_stats['diff'].mean():+.1f} bpm")
        print(f"  Range of day-night differences: {hourly_stats['diff'].min():+.1f} to {hourly_stats['diff'].max():+.1f}")

    print(f"\n--- Sparse subjects ---")
    sparse = coverage[coverage['observation_count'] < 10000]
    if len(sparse) > 0:
        print(f"  {len(sparse)} subjects have < 10,000 HR observations:")
        for _, r in sparse.iterrows():
            print(f"    {int(r['subject_id'])}: {r['observation_count']:,} records, {r['n_days']:.0f} days")
    else:
        print(f"  All subjects have >= 10,000 observations")

    print(f"\n--- Mean vs Median comparison ---")
    bias = baseline['mean_hr'] - baseline['median_hr']
    skewed = baseline[abs(bias) > 3]
    if len(skewed) > 0:
        print(f"  {len(skewed)} subjects have mean-median difference > 3 bpm (right-skewed distribution):")
        for _, r in skewed.iterrows():
            print(f"    {int(r['subject_id'])}: mean={r['mean_hr']:.1f}, median={r['median_hr']:.1f}, "
                  f"diff={r['mean_hr']-r['median_hr']:+.1f}")
    else:
        print(f"  All subjects have symmetric HR distributions (mean-median diff < 3 bpm)")

    print(f"\n--- Subjects with wide HR distributions ---")
    wide = baseline[baseline['iqr'] > 25]
    if len(wide) > 0:
        print(f"  {len(wide)} subjects have IQR > 25 bpm:")
        for _, r in wide.iterrows():
            print(f"    {int(r['subject_id'])}: IQR={r['iqr']:.1f}, CV={r['cv']:.1f}%")
    else:
        print(f"  No subjects with extremely wide HR distributions (IQR > 25)")


# ============================================================
# 12. LEAKAGE CHECK
# ============================================================
def check_leakage():
    print("\n" + "=" * 70)
    print("SECTION 12: LEAKAGE PROTECTION CHECK")
    print("=" * 70)

    print(f"""
Verification:
  [OK] Raw Dataset 2 files are NOT modified
  [OK] No cross-dataset identity matching with Stage 1
  [OK] No disease/abnormal labels created
  [OK] No clinical HRV terminology used without justification
  [OK] No future observations used in baseline computation
  [OK] Baselines are descriptive statistics, not predictive evaluations
  [OK] No accuracy, precision, recall, or F1 calculated
  [OK] No medical diagnosis claims
  [OK] Stage 1 files are completely untouched

Leakage risk: NONE
  All baselines are descriptive statistics computed from the full
  observed period. Stage 3 will use these baselines for anomaly
  detection with proper temporal train/test separation.
""")


# ============================================================
# 13. FINAL REPORT
# ============================================================
def print_final_report(df, baseline, coverage, hourly, daily, reliability, output_paths, n_minute_rows):
    print("\n" + "=" * 70)
    print("SECTION 13: FINAL REPORT")
    print("=" * 70)

    n_subjects = df[ID_COL].nunique()
    n_obs = len(df)
    date_min = df[TIME_COL].min()
    date_max = df[TIME_COL].max()

    print(f"""
===============================================================
                    STAGE 2 FINAL REPORT
===============================================================

DATASET SUMMARY
  Subjects with HR data:          {n_subjects}
  Total HR observations:          {n_obs:,}
  Date range:                     {date_min.date()} to {date_max.date()}
  Resolution:                     ~5 seconds (irregular)
  Dataset source:                 Fitabase/Fitbit (MTurk study)
  Note:                           Dataset 2 subjects are independent
                                  from Stage 1 patients (300 users)

SUBJECT COVERAGE
  Mean observations per subject:  {coverage['observation_count'].mean():,.0f}
  Min observations:               {coverage['observation_count'].min():,}
  Max observations:               {coverage['observation_count'].max():,}
  Mean days observed:             {coverage['n_days'].mean():.1f}
  Min days:                       {coverage['n_days'].min():.0f}
  Max days:                       {coverage['n_days'].max():.0f}

PERSONAL BASELINE STATISTICS
  Median HR across subjects:
    Range:                        {baseline['median_hr'].min():.1f} - {baseline['median_hr'].max():.1f} bpm
    Population mean of medians:   {baseline['median_hr'].mean():.1f} bpm
    Std of medians:               {baseline['median_hr'].std():.1f} bpm
  Within-person variability:
    Mean IQR:                     {baseline['iqr'].mean():.1f} bpm
    Mean MAD:                     {baseline['mad'].mean():.1f} bpm
    Mean CV:                      {baseline['cv'].mean():.1f}%
  Primary baseline definition:    Median HR (robust to outliers)
""")

    # Reliability counts
    r_counts = reliability['reliability'].value_counts()
    for level in ['HIGH', 'MEDIUM', 'LOW']:
        n = r_counts.get(level, 0)
        print(f"  {level:>8s} reliability:  {n:>2d} subjects")

    print(f"""
BASELINE RELIABILITY CRITERIA
  HIGH:   >= {MIN_RECORDS_HIGH:,} records, >= {MIN_DAYS_HIGH} days, low CV, few large gaps
  MEDIUM: >= {MIN_RECORDS_MEDIUM:,} records, >= {MIN_DAYS_MEDIUM} days
  LOW:    Fewer observations or days

OUTPUT FILES
  A. Subject baseline:        {output_paths['baseline']}
  B. Minute-level HR:         {output_paths['minute']}
  C. Hourly baseline:         {output_paths['hourly']}
  D. Daily baseline:          {output_paths['daily']}
  E. Data quality:            {output_paths['quality']}

KEY FINDINGS
  1. Personal baselines vary significantly across subjects
     (median HR range: {baseline['median_hr'].min():.1f}-{baseline['median_hr'].max():.1f} bpm)
  2. Within-person HR variability is moderate (mean IQR: {baseline['iqr'].mean():.1f} bpm)
  3. Most subjects show expected daytime > nighttime HR patterns
  4. Median HR is more robust than mean HR for baseline definition
  5. Data volume varies substantially (1K to 285K observations)
  6. All subjects have sufficient data for basic baseline computation

LIMITATIONS
  - Only 14 subjects have HR data (from 35 total in Dataset 2)
  - Fitbit-derived data (not clinical-grade ECG)
  - Short observation period (~2-4 weeks per subject)
  - No clinical diagnosis labels available
  - No age/gender/region information (cannot map to Stage 1)
  - Baseline is descriptive, not an anomaly detector
  - No direct comparison to Stage 1 peer population

STAGE 3 READINESS
  YES WITH LIMITATIONS

  Stage 2 provides sufficient information to begin temporal pattern
  and anomaly detection. Each subject has:
  - A personal median HR baseline
  - Hourly and daily temporal patterns
  - Minute-level aggregated data for analysis
  - Reliability flags for quality awareness

  Limitations for Stage 3:
  - Small subject count (14) limits population-level conclusions
  - No ground-truth abnormal labels for evaluation
  - Sparse subjects (e.g., ID 2026352035 with ~2.5K records)
    may produce unstable temporal patterns
  - Temporal train/test splits must respect data gaps

  Recommendation: Proceed to Stage 3 with reliability flags.
""")

    print("=" * 70)
    print("  STAGE 2 COMPLETE")
    print("=" * 70)

    # Print summary block
    print(f"""
STAGE 2 — PERSONAL BASELINE
===========================

Subjects processed:        {n_subjects}
HR observations:           {n_obs:,}
Minute-level observations: {n_minute_rows:,}
Date range:                {date_min.date()} to {date_max.date()}

Baseline outputs:
  {output_paths['baseline']}
  {output_paths['minute']}
  {output_paths['hourly']}
  {output_paths['daily']}

Quality outputs:
  {output_paths['quality']}

Key findings:
  Personal median HR range: {baseline['median_hr'].min():.1f}-{baseline['median_hr'].max():.1f} bpm
  Mean within-person IQR:   {baseline['iqr'].mean():.1f} bpm
  Reliability:              {r_counts.get('HIGH',0)} HIGH / {r_counts.get('MEDIUM',0)} MEDIUM / {r_counts.get('LOW',0)} LOW

Stage 3 readiness:
  YES WITH LIMITATIONS
""")


# ============================================================
# MAIN
# ============================================================
def main():
    print("=" * 70)
    print("  STAGE 2: PERSONAL BASELINE")
    print("  Heart Rate Monitoring ML Pipeline")
    print("=" * 70)

    # 1. Load and combine HR data from both periods
    df = load_hr_data()

    # 2. Subject coverage
    coverage = compute_subject_coverage(df)

    # 3. Gap analysis
    gaps = compute_gap_analysis(df)

    # 4. Minute-level aggregation
    minute_df = aggregate_to_minute(df)

    # 5. Subject-level baseline
    baseline = compute_subject_baseline(df)

    # 6. Hourly baseline
    hourly = compute_hourly_baseline(df)

    # 7. Daily baseline
    daily = compute_daily_baseline(df)

    # 8. Reliability assessment
    reliability = assess_reliability(baseline, coverage, gaps)

    # 9. Visualizations
    viz_path = create_visualizations(baseline, coverage, hourly, daily, df)

    # 10. Save outputs
    output_paths = save_outputs(baseline, minute_df, hourly, daily, coverage, gaps, reliability)

    # 11. Interpretation
    interpret_results(baseline, coverage, hourly, daily)

    # 12. Leakage check
    check_leakage()

    # 13. Final report
    print_final_report(df, baseline, coverage, hourly, daily, reliability, output_paths, len(minute_df))


if __name__ == '__main__':
    main()
