-- Analysis queries (SQLite). Run all with: python build_db.py --run-queries
-- name: data_coverage
SELECT station,
       COUNT(*) AS hours,
       ROUND(100.0 * COUNT(pm25) / COUNT(*), 1) AS pm25_pct_observed,
       MIN(datetime) AS first_hour, MAX(datetime) AS last_hour
FROM readings GROUP BY station ORDER BY pm25_pct_observed;

-- name: monthly_mean_pm25
SELECT substr(datetime, 1, 7) AS month, ROUND(AVG(pm25), 1) AS mean_pm25, ROUND(AVG(ws), 2) AS mean_wind
FROM readings GROUP BY month ORDER BY month;

-- name: hour_of_day_profile
SELECT CAST(substr(datetime, 12, 2) AS INTEGER) AS hour, ROUND(AVG(pm25), 1) AS mean_pm25
FROM readings GROUP BY hour ORDER BY hour;

-- name: worst_station_days
SELECT station, substr(datetime, 1, 10) AS day, ROUND(AVG(pm25), 1) AS daily_mean_pm25
FROM readings GROUP BY station, day HAVING COUNT(pm25) >= 18
ORDER BY daily_mean_pm25 DESC LIMIT 10;

-- name: share_of_days_above_good_by_station
WITH d AS (
  SELECT station, substr(datetime, 1, 10) AS day, AVG(pm25) AS m
  FROM readings GROUP BY station, day HAVING COUNT(pm25) >= 18)
SELECT station, ROUND(100.0 * SUM(m > 30) / COUNT(*), 1) AS pct_days_above_30
FROM d GROUP BY station ORDER BY pct_days_above_30 DESC;

-- name: mae_by_model_and_horizon
SELECT model, horizon, ROUND(AVG(ABS(y_pred - y_true)), 2) AS mae, COUNT(*) AS n
FROM predictions GROUP BY model, horizon ORDER BY horizon, mae;

-- name: skill_vs_persistence_by_model
WITH m AS (SELECT model, AVG(ABS(y_pred - y_true)) AS mae FROM predictions GROUP BY model)
SELECT model, ROUND(mae, 2) AS mae,
       ROUND(1 - mae / (SELECT mae FROM m WHERE model = 'Persistence'), 3) AS skill_vs_persistence
FROM m ORDER BY mae;

-- name: best_model_per_station
WITH s AS (SELECT station, model, AVG(ABS(y_pred - y_true)) AS mae FROM predictions GROUP BY station, model),
     r AS (SELECT *, RANK() OVER (PARTITION BY station ORDER BY mae) AS rk FROM s)
SELECT station, model AS best_model, ROUND(mae, 2) AS mae FROM r WHERE rk = 1 ORDER BY station;

-- name: alert_quality_above_good
-- Event = PM2.5 > 30 ug/m3 (above the CPCB "Good" band). For the Mumbai OpenAQ 8039 data the max is ~54, so the usual
-- "Poor or worse" (> 30) never occurs; change the three thresholds below if you use data with higher concentrations.
-- Precision/recall of each model at 6 h ahead.
SELECT model,
       SUM(y_pred > 30 AND y_true > 30) AS tp, SUM(y_pred > 30 AND y_true <= 30) AS fp, SUM(y_pred <= 30 AND y_true > 30) AS fn,
       ROUND(1.0 * SUM(y_pred > 30 AND y_true > 30) / NULLIF(SUM(y_pred > 30), 0), 2) AS precision,
       ROUND(1.0 * SUM(y_pred > 30 AND y_true > 30) / NULLIF(SUM(y_true > 30), 0), 2) AS recall
FROM predictions WHERE horizon = 6 GROUP BY model ORDER BY model;
