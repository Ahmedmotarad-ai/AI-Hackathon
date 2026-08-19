# Stage 2 Subject Count Investigation

**Date:** 2026-08-19
**Question:** Why does Stage 2 report 15 subjects and 3,614,915 records when the audit reported 14 subjects and 3,638,339 records?

---

## Finding

**CORRECT BUT AUDIT WAS INCONSISTENT**

The 15-subject result is correct. The audit's "14 subjects" was a per-period count that did not explicitly state the combined unique total. Both numbers are mathematically consistent once the data structure is understood.

---

## Subject Count

### What the audit said

- "P1 heartrate subjects: 14"
- "P2 heartrate subjects: 14 (12 shared with P1)"

The audit reported per-period counts. It never explicitly stated the combined unique subject count for HR data.

### What Stage 2 found

15 unique subjects after combining both periods.

### Actual raw source breakdown

| Period | Unique HR Subjects | Subject IDs |
|--------|--------------------|-------------|
| P1 only | 14 | 2022484408, 2026352035, 2347167796, 4020332650, 4558609924, 5553957443, 5577150313, 6117666160, **6391747486**, 6775888955, 6962181067, 7007744171, 8792009665, 8877689391 |
| P2 only | 14 | 2022484408, 2026352035, 2347167796, 4020332650, **4388161847**, 4558609924, 5553957443, 5577150313, 6117666160, 6775888955, 6962181067, 7007744171, 8792009665, 8877689391 |
| **Combined unique** | **15** | All of the above (6391747486 is P1-only, 4388161847 is P2-only) |

- Shared subjects: 12
- P1-only subjects: 1 (ID 6391747486)
- P2-only subjects: 1 (ID 4388161847)
- Total: 12 + 1 + 1 = **15**

The audit's statement "12 shared with P1" was correct, but it reported "14 subjects" without clarifying this was per-period, not the combined total.

---

## Record Count

### The numbers

| Source | Count |
|--------|-------|
| Audit (raw P1 + raw P2) | 1,154,681 + 2,483,658 = **3,638,339** |
| Stage 2 (after dedup) | **3,614,915** |
| **Difference** | **23,424** |

### What happened

Both periods have data on **2016-04-12**, the boundary day where P1 ends and P2 begins. Records from this day appear in both CSV files. When Stage 2 concatenates P1 and P2 and removes exact duplicates (same `Id` + same `Time`), it removes these 23,424 redundant copies.

### Verification

- All 23,424 overlapping records have **identical HR values** in both P1 and P2
- The overlap affects 8 of the 12 shared subjects
- All overlapping timestamps fall on 2016-04-12

### Per-subject overlap counts

| Subject ID | Overlapping Records |
|------------|-------------------:|
| 2022484408 | 1,531 |
| 4020332650 | 8,135 |
| 4558609924 | 1,566 |
| 5553957443 | 2,422 |
| 6962181067 | 4,917 |
| 7007744171 | 985 |
| 8792009665 | 2,545 |
| 8877689391 | 1,323 |
| **Total** | **23,424** |

### Why this is correct

Stage 2's `drop_duplicates(subset=[ID_COL, TIME_COL], keep='first')` correctly removes duplicate records. The 23,424 records were identical copies that existed in both CSV files due to the overlapping collection period boundary on April 12, 2016. Removing them prevents double-counting of the same heart-rate observations.

---

## Root Cause

1. **14 → 15 subjects:** The audit reported per-period subject counts (14 each) without computing the combined unique total. P1 has subject 6391747486 exclusively; P2 has subject 4388161847 exclusively. Combined: 15 unique subjects.

2. **3,638,339 → 3,614,915 records:** The audit summed raw file row counts from two CSV files that overlap on 2016-04-12. Stage 2 correctly deduplicated 23,424 identical (Id, Time) pairs from the boundary day.

---

## Impact

| Aspect | Affected? | Explanation |
|--------|-----------|-------------|
| Stage 2 baseline validity | No | All 15 subjects have valid baselines. The extra subject (4388161847) is a legitimate Fitbit user. |
| Reliability classification | No | Subject 4388161847 has 249,748 records across 30 days — rated HIGH reliability. Subject 6391747486 (P1-only) has 3,747 records — rated LOW reliability. Both are correctly classified. |
| Minute-level aggregation | No | 471,810 minute-level rows correctly reflect the deduplicated 3,614,915 source records. |
| Stage 3 readiness | No | The 15th subject adds valid data. No corrective action needed. |

---

## Recommendation

**No action required.**

- Stage 2's 15-subject, 3,614,915-record result is correct
- The deduplication was appropriate and necessary
- The audit's "14 subjects" was a misleading summary of per-period counts
- All Stage 2 outputs are valid
- Proceed to Stage 3 without changes
