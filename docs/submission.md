# MIND leaderboard submission reference

The [Phase 3 workflow](../README.md#34-phase-3-train-on-all-labeled-data-and-build-the-leaderboard-submission) trains the maximum-data model and scores every supplied hidden-test candidate. This reference describes the recency adjustment applied before those scores become submission ranks.

## Article age and the recency clock

MIND does not provide publication timestamps, so “age” is an exposure-age proxy. `build_item_age` scans the candidate lists in Large Train, Dev, and Test behaviors—never their click labels—and records each news ID's earliest observed candidate-impression timestamp. That first observable appearance starts the clock.

An article already in circulation when Large Train begins is therefore assigned age zero at its first candidate appearance inside the dataset; earlier history mentions do not start the clock, and the system cannot recover how long the article existed before the observation window. An ID absent from the age index, or an impression with an unparseable timestamp, falls back to age zero.

At each scored impression, age is `max(0, impression_time - first_seen_time)` in hours, capped at 720 hours and stored as `log1p(age_hours)`. Within that impression, the youngest candidate gets freshness near `+1`, the oldest near `-1`, and tied ages share a rank. The resulting submission score is `zscore(ranker_logit) + 0.02 * freshness_percentile`. Thus age is calculated at scoring time relative to each impression, rather than once relative to the start or end of the dataset.

## Submission ranks and evaluation

The adjusted scores determine the ranks written to `prediction.txt`. See [MIND submission evaluation](metrics.md#mind-submission-evaluation) for the line format and ranking metrics.

To write the candidate-attention model without the recency adjustment, run:

```powershell
python -m mindrec.cli write_submission --config configs/mind_large_submission_mpnet_candidate_attention.yaml
```

That path does not require `build_item_age`. Optional rank fusion is covered in the [ensembling guide](ensembling.md).
