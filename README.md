# changes to make:

## 1. models & app - reframe policty simulator to structural benchmark
- train xgboost on structural features only (gdp pc, growth, urbanization, tertiary enrollment, coal share, consumption pc)
- compute per-country structural_gap_pp for the most recent year & ass to country explorer as a metric card
- reframe "countries w this profile...". move the benchmark, labeled as scenario, asociational note on page
- report the structural model's own test R² in the sidebar tooltip

### goal
 sliders visibly move predictions - every country shows a gap - e.g., norway stronlgy +ve & gcc strongly -ve


## 2. backtesting3 
- rolling training & testing to check prophet's accuracy per country. train 2013 & predict 2014-2016. train 2016 & predict 2017-2019 and so on
- store per country MAE by horizon (1y/3y/5y) and 80% interval coverage to cache
- country explorer forecast tab gains one line: "Historical 3-year error for
{country}: ±X.X pp · interval coverage Y%
- same runs on the XGBoost temporal split for the paper's eval section

### goal
every priority market shows a historical-error line; results table exportable for the white paper

3. llm layer 
- depends on niya's answer

4. news retrieval
