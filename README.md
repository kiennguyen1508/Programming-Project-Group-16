# Gold and its Drivers: Economic & Finance Dashboard

**Main question:** What moves the price of gold, and has its behaviour changed since 2025?

This interactive dashboard looks at gold alongside silver, oil, the S&P 500 and several economic factors, using daily data from 2015 to October 2026. Each panel answers one part of the main question and compares 2015–2024 with the period since 2025.

## Quick start

You need Python 3.12 or newer (check with `python3 --version`). Tested with Python 3.14.7 on macOS.

**1. Download the code.** Clone the repository and move into the project folder.
```bash
git clone https://github.com/kiennguyen1508/Programming-Project-Group-16.git
cd Programming-Project-Group-16
```

**2. Set up a virtual environment.** This keeps the project's packages separate from anything else installed on your computer.
```bash
python3 -m venv .venv
source .venv/bin/activate
```

On Windows, use `python -m venv .venv` and then `.venv\Scripts\activate`.
If activation worked, your terminal prompt should start with `(.venv)`.

**3. Install the packages.** This installs everything listed in `requirements.txt`.
```bash
pip install -r requirements.txt
```

**4. Run the dashboard.**
```bash
python -m streamlit run app.py
```
Using `python -m` runs the Streamlit installation in the active virtual environment. The dashboard opens at [http://localhost:8501](http://localhost:8501). If that port is busy, Streamlit selects another port and prints its address in the terminal.

Price and macro data are stored in `data/`. The Distribution & Risk panel also uses raw trading-volume history from Yahoo Finance. Run the app while online to populate its local cache; later runs can use the cache offline when it covers the requested dates.

## Using the dashboard

The dashboard opens on an overview page with a preview of all four panels. Select **Open →** to view a panel and its controls, then **← Back to overview** to return. Each panel includes a **Method and Limitations** section describing its calculations and caveats.

## Data

All data is daily, from 2 January 2015 to 30 October 2026. It comes from Yahoo Finance, FRED (Federal Reserve Bank of St. Louis) and Cboe.

| File | What it contains | Source |
|---|---|---|
| `data/prices.csv` | Closing prices of gold futures (`GC=F`), silver futures (`SI=F`), Brent crude futures (`BZ=F`) and the S&P 500 (`^GSPC`). Only days when all four traded are kept. | Yahoo Finance |
| `data/macro.csv` | US dollar index (`DX-Y.NYB`), copper futures (`HG=F`), a broad commodity ETF (`DBC`), the 10-year real yield (`DFII10`) and the VIX. Kept separate from the prices so that adding it did not change any panel's existing results. | Yahoo Finance, FRED, Cboe |
| `data/VIX_History.csv` | Daily VIX history, used to build the VIX column in `macro.csv`. | Cboe |
| `data/volume_history.csv` | Cached daily raw trading volumes used to calculate trailing, within-instrument percentiles. Created the first time the Distribution & Risk panel loads while online. | Yahoo Finance |
| `data/volume_history_sources.json` | Labels identifying the futures volume series or ETF proxy used for each asset. | Yahoo Finance |

### Refreshing the data

This is optional and needs an internet connection. It overwrites the CSV files with the latest data.

```bash
python data_loader.py           # re-downloads prices.csv
python data_loader.py macro     # re-downloads macro.csv
```

The risk panel refreshes its volume cache when saved history does not cover the analysis periods or a source is unavailable. To force a refresh for the same dates, remove `data/volume_history.csv` and `data/volume_history_sources.json`, then restart the app while online. The older `volume_percentiles` cache files, if present, contain full-sample ranks and are no longer used. To run the project on another computer without internet, copy the new volume cache files along with the repository's existing `data/` files.

### Using the data in a panel

`data_loader.py` provides shared price, return and macro-data functions. Panels may apply different return definitions or additional date and missing-data filters to suit their analyses.

| Function | What it returns |
|---|---|
| `get_prices()` | Daily closing prices, one column per asset |
| `get_returns()` | Daily log returns of the four assets |
| `get_macro()` | Levels of the macro series |
| `get_macro_changes()` | Daily changes of the macro series: log returns for prices, basis points for the real yield, points for the VIX |
| `SPLIT` | The date used to split "before" and "since 2025" (`2025-01-01`) |

## Project structure

```
Programming-Project-Group-16/
├── app.py                    # puts the four panels on one page
├── data_loader.py            # downloads, saves and loads the shared data
├── requirements.txt          # packages to install
├── .streamlit/config.toml    # dashboard theme
├── data/                     # saved CSV files
├── panels/
│   ├── __init__.py           # lets app.py import the panels
│   ├── Charlie.py            # regression
│   ├── Kien.py               # performance and ratios
│   ├── Matthew.py            # correlation heatmap
│   └── Michael.py            # distribution and risk
├── charlie_regression.ipynb  # notebook used while building the regression panel
└── assets/                   # screenshot used in this README
```

## The panels
Each panel has its own "Method and Limitations" section in the app with the details.

### Panel 1: Regression analysis of silver, oil and gold (Charlie O'Sullivan)
**Question:** How did silver respond to gold during the 2025 boom and 2026 crash? Did silver behave more like a precious or industrial metal, and did the oil–gold relationship change in 2026 amid tensions involving Iran?

**Link to the main question:** Gold is the explanatory variable in every regression, so each β shows how silver or oil moved with gold, and whether that changed after 2025.

**Data:** Daily log returns for gold, silver and Brent oil from `prices.csv`, and copper from `macro.csv`, loaded through `data_loader.py`. Silver rose 288% from the start of 2025 to its peak on 26 January 2026, then fell 57%. The panel splits the data at that peak (Before: 2015–2024; Boom: January 2025 to the peak; Crash: after the peak) or by calendar year (2015–2024, 2025 and 2026 YTD).

**Data preprocessing:** How the raw prices are turned into the data the regressions use.
1. **Raw data.** Daily closing prices for gold (`GC=F`), silver (`SI=F`), Brent crude (`BZ=F`) and the S&P 500 are downloaded from Yahoo Finance by `data_loader.py` and saved in `data/prices.csv`. Only dates when all four assets traded are retained. Copper futures (`HG=F`) come from `data/macro.csv`.
2. **Prices to returns.** `data_loader.py` converts prices to daily log returns, ln(price today) − ln(price yesterday). The panel loads gold, silver and oil returns with `get_returns()`, and copper returns with `get_macro_changes()`. Regressing trending price levels can produce spurious relationships; returns reduce this problem. With returns expressed consistently, β estimates the change in the dependent asset's return associated with a one-unit change in gold's return.
3. **Joining copper.** Copper's daily returns are joined onto the other returns by date. Any day missing a value for the assets in a regression is dropped with `dropna()`, which is why the Silver vs Gold + Copper regressions have a few fewer days (for example 2,508 before 2025, against 2,512 for Silver vs Gold).
4. **Finding silver's peak.** The boom and crash split is calculated from `get_prices()` as silver's highest close since 1 January 2025 (26 January 2026 in the saved data), so it can update when the data is refreshed. The same prices determine the rise and fall shown under the panel title.
5. **Splitting into periods.** pandas date slicing (`.loc[start:end]`) cuts the returns into Before (2015-2024), Boom (Jan 2025 to the peak) and Crash (after the peak), or into calendar years (2015-2024, 2025, 2026 YTD).
6. **Regression inputs.** For each period, returns are passed to `run_ols`, which adds a column of ones for the intercept and estimates the model using a matrix formula. The results (β, standard error, confidence interval, t-statistic, R² and observation count) feed the panel's charts and text.
7. **For the charts.** In the scatter and 3D charts, returns are multiplied by 100 so the axes show daily % moves. This does not change β, because both sides of the regression are scaled the same way.

**What it shows:** Three regressions, each run separately for every chosen period.
- **Silver vs Gold:** silver's average daily move for each 1% daily move in gold (β). β above 1 means silver's moves were bigger than gold's.
- **Silver vs Gold + Copper:** a multiple regression with one β for each metal, holding the other constant, to see whether silver tracked gold (precious) or copper (industrial) more closely.
- **Oil vs Gold:** whether oil and gold moved in the same direction, opposite directions, or had no clear link.

The OLS estimates are calculated by hand with the matrix formula and checked against statsmodels in `charlie_regression.ipynb`. Each relationship has a scatter with fitted lines, a dot and whisker chart of each β with its confidence interval, and, for the copper regression, a rotatable 3D plane. Under the chart, a blue box answers the question in plain words, followed by the full results table (downloadable as a CSV). "What the results mean" explains each β, tests whether β changed significantly between periods (a z-test) and ends with a practical takeaway.

**Controls:** Relationship, period scheme (boom/crash or calendar years), selected periods, chart type, confidence level (90%, 95% or 99%) and a toggle for White robust standard errors.

**Key findings:**
- Silver moved more than one-for-one with gold in every period: β was 1.51 before 2025, 1.30 in the boom and 1.93 in the crash. The rise from boom to crash was significant at the 95% level (z = 2.08) but not at 99%, so silver's daily moves were largest relative to gold's after the peak.
- Holding copper constant, silver tracked gold more closely than copper in every period (β on gold 1.41, 1.15 and 1.60, against 0.26, 0.27 and 0.65 on copper), so it behaved more like a precious metal even during the boom.
- Oil and gold had a small positive link from 2015 to 2024 (β = 0.23), no clear link in 2025, and a negative link in 2026 (β = -0.37, 95% CI -0.72 to -0.03). The change from 2025 to 2026 was significant (z = -2.38), but gold explained only about 3% of oil's daily moves, so the link was weak.
- Robust standard errors matter here: in the crash, silver's standard error on gold is 0.28 with them and 0.09 without, because volatility after the peak was far from constant.

### Panel 2: Performance and ratios (Kien Nguyen)
**Question:** How has gold performed against the other assets, and does it protect investors when markets are stressed?

**How it answers the main question:** If gold is a safe haven, it should hold up on fear days and fall less than other assets in bad periods. This panel tests that directly and compares the period since 2025 with the full 2015-2026 history.

#### Data handling and transformation
**1. Data used.** Gold, silver, Brent oil and the S&P 500 come from `prices.csv` through `get_prices()`: 2,972 trading days from 2 Jan 2015 to 30 Oct 2026, with no missing values. Copper and the VIX come from `macro.csv` through `get_macro()`.

**2. Joining copper without losing price days.** Copper is added with a *left join* on the price dates. An inner join would remove gold, silver and oil dates whenever copper has no price. With a left join, those dates are retained and copper is blank; missing copper observations are excluded only when calculating the Gold/Copper ratio. No missing price is filled in.

**3. Simple returns instead of log returns.** The regression and risk panels use log returns for price assets, while the correlation panel also uses percentage changes for several macro series. This panel uses simple daily returns (`pct_change`) to report investor-facing measures such as growth of 100, total return, Sharpe and Sortino. The difference matters on large moves: on 30 January 2026, silver's simple return was −31.3%, while its log return was −37.6%.

**4. Rebasing to 100.** Gold and silver have very different price scales, so their raw prices cannot share a useful comparison chart. Each price is divided by its first value in the selected period and multiplied by 100, so each line starts at 100 and shows relative growth.

**5. Annualising.** Annualised return = (1 + total return)^(1 / years) − 1; annualised volatility = daily standard deviation × √252. For periods shorter than a year, the panel shows a warning because annualised figures can look extreme.

**6. Avoiding look-ahead bias.** Two tests use only information that would have been available at the time:
- **Stress days** use the VIX close from the *previous* trading day. Same-day VIX would be circular, because the VIX jumps on the same days stocks fall.
- **The reversion test** compares each day's ratio with a *rolling* 5-year average and σ, not the full-sample average, and only starts once 2 years of history exist (from about January 2017). Using the full-sample average would mean judging 2018 with knowledge of 2026.

**7. Two different averages for the ratios.** The ratio chart's bands, z-score and percentile use the full 2015–2026 history, so the reference average stays fixed when a different period is selected. The reversion test uses the rolling average described above, reflecting the information available at each date.

**8. Extreme days retained.** Large moves, including silver's decline on 30 January 2026 and the S&P 500's +9.5% move on 9 April 2025, were checked and retained because they are relevant to the analysis. The Stress days tab offers the median as well as the average so a single extreme day has less influence on the summary.

**9. Sample size after each step.** Full period: 2,971 return days, of which 391 are stressed (VIX ≥ 25 the day before) and 2,580 calm. Since 2025: 458 return days, of which only 34 are stressed and 424 calm.

#### What you see
Three tabs:
1. **Performance:** a chart (Growth of 100, Drawdown or Replay) and a table of total return, annualised return, volatility, maximum drawdown, Sharpe ratio and Sortino ratio, with the best value in each column highlighted. A text box summarises the table.
2. **Stress days:** each asset's average (or median) daily return on calm days and on stressed days.
3. **Ratios:** the gold/silver, gold/oil and gold/copper ratios with their 2015–2026 average and ±1σ/±2σ bands, plus the latest value in the saved data, z-score and percentile. Below this, the reversion test shows how often a ratio moved back towards its average after being more than 2σ away, alongside a base rate (how often it moved that way on any day).

#### How to read it
- **Sharpe and Sortino:** return per unit of risk; higher is better. Sortino only counts days below the risk-free rate as risk.
- **Drawdown:** how far an asset is below its previous peak; closer to 0% is better.
- **Stress days:** if gold protects investors, its stressed-day bar should be at or above its calm-day bar and above the S&P 500's.
- **Reversion test:** the signal only means something if "Moved back towards mean" is clearly higher than the base rate.

#### Controls
Period (Since 2025, YTD, 1Y, 5Y, Max; default Since 2025), asset picker and a "Start from" slider. Performance tab: chart type (Growth of 100, Drawdown, Replay), log-scale toggle and risk-free rate slider (0–6%, default 4%). Stress days tab: VIX threshold (15–40, default 25) and Average/Median. Ratios tab: ratio picker and look-ahead horizon (1, 3 or 6 months). A CSV download of the prices and a "Method and Limitations" section are at the bottom of the panel.

#### Key findings
Saved data to 30 Oct 2026; the settings to reproduce each finding are in brackets.
- **Since 2025, gold earned more but became riskier.** Annualised return 23.2% with 25.1% volatility and a Sharpe ratio of 0.80, the highest of the four assets, against 10.6%, 16.8% and 0.45 over the full period. Silver and oil returned more since 2025 (32.1% and 27.2%) but with far higher volatility (52.8% and 45.1%). *[Performance, Since 2025 vs Max]*
- **Gold is in its deepest fall since 2015.** It peaked at $5,318 on 29 Jan 2026 and closed at $3,899 on 30 Oct 2026, a 26.7% drawdown. That is deeper than any fall from 2015 to 2024 (the worst was about 21%). *[Performance, Max, Drawdown]*
- **Gold did not reliably protect investors on fear days.** On the 391 stressed days, gold averaged +0.01% a day against +0.05% on calm days, while the S&P 500 averaged +0.13% on stressed days. The median gives the same ranking. *[Stress days, Max, VIX 25]*
- **Since 2025 gold did rise on stressed days, but so did everything else.** On the 34 stressed days, gold averaged +0.18% (+0.09% on calm days), but the S&P 500 averaged +0.28% and silver +0.27%. This looks like markets bouncing back after fear spikes, not gold protecting investors, and 34 days is a small sample. *[Stress days, Since 2025, VIX 25]*
- **Only the gold/silver ratio reliably moved back to its average.** After being more than 2σ from its rolling 5-year average, it moved back within 3 months 81% of the time, against a base rate of 43–57%. Gold/oil (53% vs 47%) and gold/copper (54% vs 45%) were barely better than chance. Today the gold/silver ratio is 79.4, right at its average (z = −0.04, 50th percentile), so there is no signal now. *[Ratios, 3 months]*

#### Limitations
- The "since 2025" period is under two years and has only 34 stressed days, so those results are less certain than the full-period ones.
- The reversion test counts overlapping days (259 above +2σ and 105 below −2σ for gold/silver), so the number of independent episodes is much smaller than the day counts suggest.
- Ratios can spike far upwards but cannot fall below zero, so the σ bands are only approximate. The percentile is more reliable.
- The risk-free rate is fixed for the whole period (default 4%), but cash rates were near zero until 2022, so Sharpe and Sortino over Max are only approximate.
- Futures prices jump slightly when contracts roll over, and the S&P 500 is a price index without dividends, which understates its return.
- The panel describes the past and is not a forecast.

### Panel 3: Correlation Heatmap (Matthew Herbert)
**Question:** Which assets move together with gold, and did those links change in relation to various economic events?

**Data:** The panel draws on files in the `data/` folder:
- `data/prices.csv`: Daily closing prices for Gold, Silver, Oil and the S&P 500 from Yahoo Finance, loaded through `get_returns()` in `data_loader.py`.
- `data/macro.csv`: Daily levels for commodities, the US dollar and Copper, plus the 10-year real yield and VIX. The panel loads this file with its `load_macro()` function.
- `data/DFII10.csv`: This file supplies the real yield column figures in `macro.csv`.
- `data/VIX_History.csv`: Supplies the VIX values used in `macro.csv`.

Gold, Silver, Oil and the S&P 500 use daily log returns. Commodity, dollar and VIX series use daily percentage changes; the real yield uses the daily change in percentage points. Dates with incomplete data are excluded from the panel's complete-case analysis.

**1. Loading the data.** The panel uses two saved files, so it runs without internet:
- `data/prices.csv` (2,972 trading days, 2 Jan 2015 to 30 Oct 2026): Gold, Silver, Brent Oil and the S&P 500. It is read through the shared `get_returns()` function in `data_loader.py`.
- `data/macro.csv` (2,980 days over the same dates): Commodities (DBC), the Dollar index, Copper, the 10-year real yield and the VIX. It is read directly by the panel's own `load_macro()` function, with the Date column parsed as the index.
**2. Turning prices into returns.** Correlating price levels can show misleading links between assets that both trend over time, so each series is converted to a daily change:
- Gold, Silver, Oil and the S&P 500 use daily log returns, ln(Pₜ / Pₜ₋₁). Other panels may use different return definitions for their analyses.
- Commodities, the Dollar, Copper and the VIX use the daily % change (`pct_change()`).
- The real yield is already a rate, so a % change of a yield near zero would give huge, meaningless numbers. It uses the daily change in percentage points instead (`diff()`).
The first day of each series is omitted because calculating a change requires a previous observation.
**3. Matching the dates.** The two tables are combined with an inner join, keeping only dates present in both files (2,971 days). The markets use different calendars, and `macro.csv` has gaps: 25 missing real-yield values, 5 for Copper and Commodities, 4 for the Dollar and 1 for the VIX. A missing value can also leave the following daily change blank.
**4. Removing incomplete days.** `dropna()` removes any date with a blank in any column, leaving **2,925 complete trading days** (46 removed, about 1.5%) in the saved dataset. Every correlation therefore uses the same dates, making pairs easier to compare. No values are filled in or estimated.
**5. Splitting by period.** The shared `SPLIT` date (1 Jan 2025) and the silver peak (26 Jan 2026) divide the data for the quick period buttons:
- Before: 2,475 days (Jan 2015 to Dec 2024)
- Boom: 259 days (1 Jan 2025 to 26 Jan 2026)
- Crash: 191 days (27 Jan 2026 to Oct 2026)
The Animated Timeline groups the data by calendar year, with about 245 to 249 days per year, except 2025 (243) and 2026 (207, January to October only).

**6. Preparing for the statistics.** Correlations are clipped to between -0.999 and +0.999 before the Fisher z-transform (arctanh), since a correlation of exactly ±1 would give an infinite value. The rolling correlation needs a full window before it starts, so the first 29 to 249 days (depending on the window chosen) have no value and are dropped from the graph.

**What Panel 3 shows:**
- **Correlation heatmap:** A staircase heatmap of Pearson correlations (r) for each selected asset pair. Gold indicates positive correlation and blue indicates negative correlation. Pairs that are not statistically significant at the selected threshold are greyed out and marked “n.s.”
- ***Rolling Graph:*** How the correlation between two assets changes over time, with a shaded 95% confidence band. Shaded boxes mark the COVID crash, the Ukraine war and rate hikes, and the selected period. A dotted line marks the silver peak (26 Jan 2026). Dotted average lines compare before and during the selected period, and a row of cards shows the latest, average, lowest and highest r. A second pair can be added for comparison.
- ***Animated Timeline:*** One heatmap per year from 2015 to 2026, played with a Play button or a year slider, to show how the links shift year by year. 2026 covers January to October only.
- ***Decoupling Analysis:*** Compares the selected period with all the days before it. A table ranks every pair by how much its correlation changed, with a Fisher z-test p-value for each change, and can be downloaded as a CSV. It can also be shown as a heatmap of the changes. The "Pair in Focus" section then tests one chosen pair and says in plain words whether the change is real or just random noise.

**Controls:** An assets picker; Quick Period Select buttons (Before, Boom, Crash, 2025 Onwards); a time period slider; and a significance level slider (0.01 to 0.20, default 0.05). The Rolling Graph has its own "Comparing" and "Against" dropdowns and a smoothing window slider (30 to 250 days, default 90). Controls that do nothing on a view are hidden on that view.
**Key findings and interpretations:**
- Over the whole period, gold's strongest link is with silver (r = 0.77). Gold moves against the dollar (r = -0.38) and against real yields (r = -0.30), and has almost no link with the S&P 500, oil or the VIX.
- Gold's link with real yields mostly faded after 2025. It went from -0.36 before 2025 to -0.10 since, which is a significant change (Fisher z-test, p < 0.001). The textbook "higher real yields hurt gold" link broke down during the boom.
- 25 of the 36 pairs changed significantly after 2025. The biggest changes are in oil, which almost stopped moving with the S&P 500 (0.28 to -0.09) and copper (0.28 to -0.06).
- Gold and silver stayed tightly linked throughout. Their 90-day rolling correlation never fell below 0.50, and in the crash after the silver peak it rose to 0.84, higher than in the boom (0.64).
- In the crash, gold started moving with the S&P 500 (r = 0.36, against roughly zero before) and against the VIX (r = -0.30). This suggests gold was sold off alongside stocks rather than acting as a safe haven.

**Limitations:** Correlation only measures straight-line links and says nothing about cause. Daily closing times differ across markets, which can weaken some links. Rolling correlations depend on the window chosen, and the boom and crash periods are short, so their estimates are less certain.

### Panel 4: Distribution & Risk (Michael Gilgunn)

**Question:** How has the relationship between daily price movements and trading activity evolved since 2025, and how does this compare across different assets?

**Data:** The panel uses daily price, return and trading-volume data from the project's shared data files and Yahoo Finance.

- `data/prices.csv`: Daily closing prices for Gold, Silver, Oil and the S&P 500, loaded through `get_returns()` in `data_loader.py`.
- `data/macro.csv`: Copper price data, accessed through `get_macro()` and `get_macro_changes()`.
- Yahoo Finance: Daily trading-volume data. Futures volume is used where sufficient historical coverage is available; otherwise, an ETF proxy is used: GLD (Gold), SLV (Silver), CPER (Copper), USO (Oil) or SPY (S&P 500).
- `data/volume_history_sources.json`: Records which volume source is used for each asset.

Volume data is cached locally in `data/volume_history.csv`, so it does not need to be downloaded again on every run. Raw trading volume is converted into a trailing 252-observation percentile for each instrument. Each percentile uses only observations available up to that date, avoiding look-ahead bias. A percentile of 90 means trading activity was higher than approximately 90% of the instrument's previous observations. This allows activity to be compared across markets with very different trading volumes.

**Data preprocessing:**


**1. Loading the data.** The panel loads daily log returns for Gold, Silver, Oil and the S&P 500 through `get_returns()`. Copper returns are obtained through `get_macro()` and `get_macro_changes()`. Trading-volume histories are retrieved from Yahoo Finance where necessary and saved locally for future use.

**2. Preparing trading volume.** Raw volume is converted into a trailing 252-observation percentile separately for each instrument. This standardises trading activity relative to its own historical levels rather than comparing absolute volumes across different markets.

**3. Aligning the data.** The volume percentiles are matched to the corresponding daily returns by date. This ensures that each observation represents the return and trading activity for the same trading day.

**4. Handling volume sources.** Futures volume is used where sufficient historical coverage exists. Otherwise, a related ETF is used as a proxy. The source for each asset is recorded in `volume_history_sources.json` so that the user can identify which volume series is being analysed.

**5. Splitting into periods.** The shared split date (`2025-01-01`) separates the pre-2025 and post-2024 observations. The panel also allows analysis of individual calendar years, including 2026 YTD.

**What it shows:**

- **Joint Probability Distribution:** A two-dimensional Gaussian Kernel Density Estimation (KDE) creates an interactive 3D surface. The X-axis represents daily log returns in percentage terms, the Y-axis represents the trailing volume percentile (0–100), and the Z-axis represents estimated joint probability density. Higher areas indicate combinations of returns and trading activity that occur more frequently in the selected sample.

- **Distribution Comparison:** Users can compare two different assets or periods using side-by-side 3D density surfaces and 2D contour plots. Both surfaces share a density-axis scale, while the contour plots share a colour scale, making differences in distribution shape and concentration easier to identify. Reflected KDE observations reduce estimation bias near the volume-percentile boundaries of 0 and 100.

- **Risk Measures:** A table reports observation count, mean daily return, annualised volatility, 95% Value at Risk (VaR), 95% Expected Shortfall, skewness, excess kurtosis and maximum drawdown. Annualised volatility is calculated using daily standard deviation multiplied by √252. VaR represents the fifth-percentile daily loss, while Expected Shortfall measures the average loss among observations at or below that threshold.

- **Historical VaR Backtest:** The fifth-percentile loss threshold from 2015–2024 is held fixed and tested against returns since 2025. The table reports the number and percentage of breaches against the expected 5% rate. Kupiec's test examines whether the breach rate matches the expected rate, while Christoffersen's test examines whether breaches are independent over time.

- **Rolling Volatility:** A time-series chart displays annualised rolling volatility using a selectable window of 21, 30, 60 or 90 trading days. A vertical marker identifies the beginning of 2025, allowing changes in volatility to be compared across the two periods.

- **Trading Activity and Return Magnitude:** Observations are divided into low- and high-volume terciles within 2015–2024 and since 2025. The panel compares mean absolute daily log returns between the two groups, calculates the high-minus-low difference and examines how this relationship changed after 2024. Gold's change is also compared against the average change across the available comparison assets.

- **Distribution and Variance Tests:** Kolmogorov–Smirnov tests compare the pre-2025 and post-2024 return distributions, while Levene tests examine whether their variances differ. The results include the post/pre volatility ratio, sample sizes and Holm-adjusted p-values to account for multiple hypothesis testing.

**Controls:** Asset and period selection, with the option to add a second asset or period for distribution comparisons. The KDE bandwidth can be adjusted from 0.65× to 1.4× to control the smoothing of the estimated density surface. The panel also displays the volume source used for each asset. A CSV download provides the selected assets' daily log returns and corresponding volume percentiles.

**Interpreting the results:**

- A wider distribution along the return axis indicates greater dispersion in daily price movements.
- Greater concentration towards negative returns indicates a higher concentration of downside observations.
- A positive high-minus-low difference indicates larger absolute daily returns during high-volume observations.
- A positive Gold-versus-peer change indicates that Gold's high-minus-low difference increased more, or decreased less, than the average across the comparison assets.

These comparisons are descriptive and do not establish a causal relationship between trading activity and price movements.

**Limitations:**

- ETF volume is used as a proxy where sufficient futures-volume history is unavailable. It does not represent the exact trading volume of the futures contract or index being analysed.
- Continuous futures contracts can experience volume disruptions around contract rollovers.
- Early volume percentiles use fewer than 252 observations, making initial estimates less representative of a full historical window.
- Tail-risk estimates are less stable when based on small samples; observation counts are provided in the risk table.
- The 2026 sample is incomplete and should not be treated as a full-year result.
- Tail estimates from small samples are less stable. The Kolmogorov–Smirnov, Levene, Kupiec and Christoffersen tests rely on assumptions that may not fully hold for financial returns, particularly where volatility clustering is present. Their results should therefore be treated as indicative rather than definitive.
- Historical distributions and backtesting results describe past market behaviour and cannot predict future performance.


## Contributions

| Member | Panel | Shared work |
|---|---|---|
| Charlie O'Sullivan | Regression analysis of silver, oil and gold | Built the regression panel and its period comparisons, charts and statistical summaries. |
| Kien Nguyen | Performance & ratios | Set up the repository and folder structure; built `data_loader.py` (price download, `get_prices()`, `get_returns()`) and later added the macro data (`get_macro()`, `get_macro_changes()`); built the overview grid in `app.py` and the dark theme; wrote the README apart from the other panel sections; kept `requirements.txt` up to date |
| Matthew Herbert | Correlation heatmap | Built Panel 3 (`panels/Matthew.py`), including the significance-masked heatmap, rolling correlations, animated timeline and decoupling analysis. Documented Panel 3 and contributed to dashboard layout and debugging. |
| Michael Gilgunn | Distribution & risk | Built Panel 4 (`panels/Michael.py`), including the joint-distribution visualisation, risk measures, volume handling, backtests and statistical comparisons. |

## Limitations
- Gold, silver, oil and copper are future prices, which jump slightly when contracts roll over.
- The S&P 500 is a price index, so it does not include dividends.
- The saved data stops on 30 October 2026. Run the refresh commands above to update it.
- The "since 2025" period is less than two years long, so its results are less reliable than the 2015–2024 ones.
- The dashboard shows how gold moves with other markets, not what causes those moves.
- This project was built for a university module and is not investment advice.