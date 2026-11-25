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
If it worked, your terminal prompt now starts with `(.venv)`.

**3. Install the packages.** This installs everything listed in `requirements.txt`.
```bash
pip install -r requirements.txt
```

**4. Run the dashboard.**
```bash
python -m streamlit run app.py
```
Using `python -m` makes sure the Streamlit inside the virtual environment is used, not one installed elsewhere on your computer.
The dashboard opens in your browser at http://localhost:8501.
If that port is busy, Streamlit uses the next one (8502) and prints the address in the terminal.

The shared price and macro data is saved in the `data/` folder. The Distribution & Risk panel also needs raw trading-volume history from Yahoo Finance: run the app once while online to download and save that data. Later runs use the local cache and work without internet.
## Using the dashboard

The dashboard opens on an overview page with a preview of all four panels. Click **Open →** on any panel to see it in full with its controls, and **← Back to overview** to return. Each panel has a "Method and Limitations" section explaining how its results were calculated.

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

Every panel loads its data through `data_loader.py`, so all four panels use the same dates and the same definitions.

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

###  Panel 1: Regression analysis of silver, oil and gold (Charlie O'Sullivan)
**Question:** How did silver react to gold's movements through the 2025 boom and 2026 crash, does silver behave more like a precious or an industrial metal, and did the link between oil and gold change in 2026, the year of the Iran tension?

**Link to the main question:** Gold is the explanatory variable in every regression, so each β shows how silver or oil moved with gold, and whether that changed after 2025.

**Data:** Daily log returns of gold, silver and Brent oil from `prices.csv`, and copper from `macro.csv`, all loaded through `data_loader.py`. Silver rose 288% from the start of 2025 to its peak on 26 Jan 2026, then fell 57%. The data is split either at that peak (Before: 2015-2024, Boom: Jan 2025 to the peak, Crash: after the peak) or by calendar year (2015-2024, 2025, 2026 YTD).

**Data preprocessing:** How the raw prices are turned into the data the regressions use.
1. **Raw data.** Daily closing prices of gold (`GC=F`), silver (`SI=F`) and Brent crude (`BZ=F`) futures, downloaded from Yahoo Finance by `data_loader.py` and saved in `data/prices.csv`. Only days when all four assets in that file traded are kept, so every row has a price for each asset. Copper futures (`HG=F`) come from `data/macro.csv`.
2. **Prices to returns.** `data_loader.py` turns each price series into daily log returns, ln(price today) - ln(price yesterday). The panel loads these with `get_returns()` for gold, silver and oil, and `get_macro_changes()` for copper, and caches them so they are only loaded once. Prices trend over time, so regressing one price on another can show a link that is not really there (a spurious regression). Returns avoid this, and β can then be read as the % move in silver or oil for each 1% move in gold.
3. **Joining copper.** Copper's daily returns are joined onto the other returns by date. Any day missing a value for the assets in a regression is dropped with `dropna()`, which is why the Silver vs Gold + Copper regressions have a few fewer days (for example 2,508 before 2025, against 2,512 for Silver vs Gold).
4. **Finding silver's peak.** The boom and crash split is not typed in. It is found from the price data with `get_prices()`, as silver's highest closing price since 1 January 2025 (26 Jan 2026), so it updates if the data is refreshed. The same prices give the rise and fall shown under the panel title.
5. **Splitting into periods.** pandas date slicing (`.loc[start:end]`) cuts the returns into Before (2015-2024), Boom (Jan 2025 to the peak) and Crash (after the peak), or into calendar years (2015-2024, 2025, 2026 YTD).
6. **Regression inputs.** For each period, the returns are passed to `run_ols`, which adds a column of ones for the intercept and estimates the model by matrix formula. The results (β, standard error, confidence interval, t-statistics, R² and the number of days) are collected into one table that feeds every chart and sentence in the panel.
7. **For the charts.** In the scatter and 3D charts, returns are multiplied by 100 so the axes show daily % moves. This does not change β, because both sides of the regression are scaled the same way.

**What it shows:** Three regressions, each run separately for every chosen period.
- **Silver vs Gold:** silver's average daily move for each 1% daily move in gold (β). β above 1 means silver's moves were bigger than gold's.
- **Silver vs Gold + Copper:** a multiple regression with one β for each metal, holding the other constant, to see whether silver tracked gold (precious) or copper (industrial) more closely.
- **Oil vs Gold:** whether oil and gold moved in the same direction, opposite directions, or had no clear link.

The OLS estimates are calculated by hand with the matrix formula and checked against statsmodels in `charlie_regression.ipynb`. Each relationship has a scatter with fitted lines, a dot and whisker chart of each β with its confidence interval, and, for the copper regression, a rotatable 3D plane. Under the chart, a blue box answers the question in plain words, followed by the full results table (downloadable as a CSV). "What the results mean" explains each β, tests whether β changed significantly between periods (a z-test) and ends with a practical takeaway.

**Controls:** relationship, how to split the data (boom and crash or calendar years) and which of those periods to include, chart type, confidence level (90%, 95% or 99%) and a toggle for White robust standard errors.

**Key findings:**
- Silver moved more than one-for-one with gold in every period: β was 1.51 before 2025, 1.30 in the boom and 1.93 in the crash. The rise from boom to crash was significant at the 95% level (z = 2.08) but not at 99%, so silver's daily moves were largest relative to gold's after the peak.
- Holding copper constant, silver tracked gold more closely than copper in every period (β on gold 1.41, 1.15 and 1.60, against 0.26, 0.27 and 0.65 on copper), so it behaved more like a precious metal even during the boom.
- Oil and gold had a small positive link from 2015 to 2024 (β = 0.23), no clear link in 2025, and a negative link in 2026 (β = -0.37, 95% CI -0.72 to -0.03). The change from 2025 to 2026 was significant (z = -2.38), but gold explained only about 3% of oil's daily moves, so the link was weak.
- Robust standard errors matter here: in the crash, silver's standard error on gold is 0.28 with them and 0.09 without, because volatility after the peak was far from constant.

###  Panel 2: Performance and ratios (Kien Nguyen)
**Question:** How has gold performed against the other assets, and does it protect investors when markets are stressed?

**How it answers the main question:** If gold is a safe haven, it should hold up on fear days and fall less than other assets in bad periods. This panel tests that directly and compares the period since 2025 with the full 2015-2026 history.

#### Data handling and transformation
**1. Data used.** Gold, silver, Brent oil and the S&P 500 come from `prices.csv` through `get_prices()`: 2,972 trading days from 2 Jan 2015 to 30 Oct 2026, with no missing values. Copper and the VIX come from `macro.csv` through `get_macro()`.

**2. Joining copper without losing price days.** Copper is added with a  *left join* on the price dates. An inner join would have deleted gold, silver and oil days whenever copper had no price. With a left join, those days stay in the data and copepr is simply blank, and blank copper days (at most 5) are dropped only when the Gold/Copper ratio is calculated. No missing value is ever filled in, so no prices are invented.

**3. Simple returns instead of log returns.** The other panels use log returns because they add up across days and suit regressions and correlations. This panel uses simple daily returns (`pct_change`) because it reports what an investor actually earned: growth of 100, total return, Sharpe and Sortino. The difference matters on big days: on 30 Jan 2026 silver's simple return was -31.3%, but its log return was -37.6%.

**4. Rebasing to 100.** Gold trades around $4,000 and silver around $80, so their raw prices can't share one chart. Each price is divided by its first falue in the chosen period and multiplied by 100, so every line starts at 100 and shows growth in percentage terms.

**5. Annualising.** Daily figures are converted to yearly ones using 252 trading days: annualised return = (1 + total return)^(1/years) -1, and volatility = daily standard deviation x √252. For periods shorter than a year the panel shows a warning, because annualised figures can look extreme.

**6. Avoiding look-abead bias** Two tests only use information that was available at the time:
- **Stress days** use the VIX close from the *previous* trading day. Same-day VIX would be circular, because the VIX jumps on the same days stocks fall.
- **The reversion test** compares each day's ratio with a *rolling* 5-year average and σ, not the full-sample average, and only starts once 2 years of history exist (from about January 2017). Using the full-sample average would mean judging 2018 with knowledge of 2026.

**7. Two different averages for the ratios, on purpose.** The ratio chart's band, z-score and percentile use the full 2015-2026 history, so the "average" doesn't change everytime you click a different period. The reversion test uses the rolling average from point 6, because it simulates what an investor would have known at the time.

**8. Extreme days kept.** Large moves, such as silver's crash on 30 Jan 2026 and the S&P 500's +9.5% on 9 Apr 2025, were checked and kept, because they are exactly the events the panel studies. To stop one day dominating the result, the Stress days tab offers the median as well as the average.

**9. Sample size after each step.** Full period: 2,971 return days, of which 391 are stressed (VIX ≥ 25 the day before) and 2,580 calm. Since 2025: 458 return days, of which only 34 are stressed and 424 calm.

#### What you see
Three tabs:
1. **Performance:** a chart (Growth of 100, Drawdown or Replay) and a table of total return, annualised return, volatility, maximum drawdown, Sharpe ratio and Sortino ratio, with the best value in each column highlighted. A text box summarises the table.
2. **Stress days:** each asset's average (or median) daily return on calm days and on stressed days.
3. **Ratios:** the gold/silver, gold/oil and gold/copper ratios with their 2015–2026 average and ±1σ/±2σ bands, plus today's value, z-score and percentile. Below this, the reversion test shows how often a ratio moved back towards its average after being more than 2σ away, next to a base rate (how often it moved that way on any day).

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
- The "since 2025" period is under two years and has only 34 stressed days, so those results are much less certain than the full-period ones.
- The reversion test counts overlapping days (259 above +2σ and 105 below −2σ for gold/silver), so the number of independent episodes is much smaller than the day counts suggest.
- Ratios can spike far upwards but cannot fall below zero, so the σ bands are only approximate. The percentile is more reliable.
- The risk-free rate is fixed for the whole period (default 4%), but cash rates were near zero until 2022, so Sharpe and Sortino over Max are only approximate.
- Futures prices jump slightly when contracts roll over, and the S&P 500 is a price index without dividends, which understates its return.
- The panel describes the past and is not a forecast.

###  Panel 3: Correlation Heatmap (Matthew Herbert)
**Question:** Which assets move together with gold, and did those links change in relation to various economic events?

**Data:** My panel draws on four files in the `data` folder:
- `data/prices.csv`: This is the daily closing prices for Gold, Silver, Oil and the S&P 500, from Yahoo Finance, which is called by the `get_returns()` function in `data_loader.py` python file.
- `data/macro.csv` : This is the daily levels of Commodities, the Dollar and Copper also from Yahoo Finance, the 10-year yield and the VIX, called by `load_macro()` in my panel.
- `data/DFII10.csv`: This file supplies the real yield column figures in `macro.csv`.
- `data/VIX_History.csv` : This supplies the VIX column figures in `macro.csv`. 

Gold, Silver, Oil and the S&P 500 use daily log returns. The other assets use the daily percentage change, apart from the real yield, which uses the daily change in percentage points. In certain trading days, there is not complete data on each of the analysed and used assets. In these cases, the trading day as a whole is excluded, leaving apporximately 
2,900 complete trading days from Jan 2015, to October 2026, which my panel uses.

**Data Preprocessing:**
**1. Loading the data.** The panel uses two saved files, so it runs without internet:
- `data/prices.csv` (2,972 trading days, 2 Jan 2015 to 30 Oct 2026): Gold, Silver, Brent Oil and the S&P 500. It is read through the shared `get_returns()` function in `data_loader.py`.
- `data/macro.csv` (2,980 days over the same dates): Commodities (DBC), the Dollar index, Copper, the 10-year real yield and the VIX. It is read directly by the panel's own `load_macro()` function, with the Date column parsed as the index.
**2. Turning prices into returns.** Correlating price levels would show false links between any two assets that both trended upwards over time, so every series is first turned into a daily change:
- Gold, Silver, Oil and the S&P 500 use daily log returns, ln(Pₜ / Pₜ₋₁), the same definition every panel uses.
- Commodities, the Dollar, Copper and the VIX use the daily % change (`pct_change()`).
- The real yield is already a rate, so a % change of a yield near zero would give huge, meaningless numbers. It uses the daily change in percentage points instead (`diff()`).
The first day of each series is lost, because a change needs a previous day.
**3. Matching the dates.** The two tables are combined with an inner join, so only dates found in both files are kept (2,971 days). The markets do not share an identical calendar, and `macro.csv` has a few gaps: 25 missing real yield values (US bond market holidays), 5 for Copper and Commodities, 4 for the Dollar and 1 for the VIX. A gap blanks the change on that day and the next, so the real yield ends up with 44 blank days.
**4. Removing incomplete days.** `dropna()` removes any day with a blank in any column, leaving **2,925 complete trading days** (46 removed, about 1.5%). This means every correlation in the panel uses exactly the same days, so pairs can be compared fairly. No values are filled in or estimated. A check found no infinite values after the % changes.
**5. Splitting by period.** The shared `SPLIT` date (1 Jan 2025) and the silver peak (26 Jan 2026) divide the data for the quick period buttons:
- Before: 2,475 days (Jan 2015 to Dec 2024)
- Boom: 259 days (1 Jan 2025 to 26 Jan 2026)
- Crash: 191 days (27 Jan 2026 to Oct 2026)
The Animated Timeline groups the data by calendar year, with about 245 to 249 days per year, except 2025 (243) and 2026 (207, January to October only).
**6. Preparing for the statistics.** Correlations are clipped to between -0.999 and +0.999 before the Fisher z-transform (arctanh), since a correlation of exactly ±1 would give an infinite value. The rolling correlation needs a full window before it starts, so the first 29 to 249 days (depending on the window chosen) have no value and are dropped from the graph.

**What Panel 3 Shows:**
- ***Correlation Heatmap:*** A staircase heatmap of the Pearson correlation (r) for every pair of chosen assets. Gold means the pair moves together and blue means they move in opposite directions. Pairs that are not significant at the chosen level are greyed out and marked "n.s.", so the eye goes only to the links that are real.
- ***Rolling Graph:*** How the correlation between two assets changes over time, with a shaded 95% confidence band. Shaded boxes mark the COVID crash, the Ukraine war and rate hikes, and the selected period. A dotted line marks the silver peak (26 Jan 2026). Dotted average lines compare before and during the selected period, and a row of cards shows the latest, average, lowest and highest r. A second pair can be added for comparison.
- ***Animated Timeline:*** One heatmap per year from 2015 to 2026, played with a Play button or a year slider, to show how the links shift year by year. 2026 covers January to October only.
- ***Decoupling Analysis:*** Compares the selected period with all the days before it. A table ranks every pair by how much its correlation changed, with a Fisher z-test p-value for each change, and can be downloaded as a CSV. It can also be shown as a heatmap of the changes. The "Pair in Focus" section then tests one chosen pair and says in plain words whether the change is real or just random noise.

**Controls:** An assets picker; Quick Period Select buttons (Before, Boom, Crash, 2025 Onwards); a time period slider; and a significance level slider (0.01 to 0.20, default 0.05). The Rolling Graph has its own "Comparing" and "Against" dropdowns and a smoothing window slider (30 to 250 days, default 90). Controls that do nothing on a view are hidden on that view.
**Key findings and Interpretations:**
- Over the whole period, gold's strongest link is with silver (r = 0.77). Gold moves against the dollar (r = -0.38) and against real yields (r = -0.30), and has almost no link with the S&P 500, oil or the VIX.
- Gold's link with real yields mostly faded after 2025. It went from -0.36 before 2025 to -0.10 since, which is a significant change (Fisher z-test, p < 0.001). The textbook "higher real yields hurt gold" link broke down during the boom.
- 25 of the 36 pairs changed significantly after 2025. The biggest changes are in oil, which almost stopped moving with the S&P 500 (0.28 to -0.09) and copper (0.28 to -0.06).
- Gold and silver stayed tightly linked throughout. Their 90-day rolling correlation never fell below 0.50, and in the crash after the silver peak it rose to 0.84, higher than in the boom (0.64).
- In the crash, gold started moving with the S&P 500 (r = 0.36, against roughly zero before) and against the VIX (r = -0.30). This suggests gold was sold off alongside stocks rather than acting as a safe haven.

**Limitations:** Correlation only measures straight-line links and says nothing about cause. Daily closing times differ across markets, which can weaken some links. Rolling correlations depend on the window chosen, and the boom and crash periods are short, so their estimates are less certain.

### Panel 4: Distribution & Risk (Michael Gilgunn)
**Question:** How has the relationship between daily price movements and trading activity evolved since 2025, and how does this compare across different assets?

**Code and data:** The panel is implemented in `panels/Michael.py` and loaded by `app.py`. It uses Streamlit for the interface, Pandas and NumPy for data processing, Plotly for charts, SciPy for statistical analysis and `yfinance` to retrieve trading volume.

- `data/prices.csv` provides daily prices for Gold, Silver, Oil and the S&P 500. The panel loads their daily log returns through `get_returns()` in `data_loader.py`.
- `data/macro.csv` provides the Copper price series. The panel obtains Copper returns through `get_macro()` and `get_macro_changes()`, using the same log-return basis as the other assets.
- Daily volume comes from Yahoo Finance. Futures volume is used when the data has sufficient coverage; otherwise, the panel uses a labelled ETF proxy: GLD, SLV, CPER, USO or SPY. `data/volume_history_sources.json` records the source used for each asset.

Volume histories are cached locally in `data/volume_history.csv`, so once downloaded they can be reused without repeatedly contacting Yahoo Finance. The panel converts each instrument's raw volume into a trailing 252-observation percentile, using only observations available up to that date. A percentile near 90 indicates relatively high activity for that instrument; it does not mean the same number of shares or contracts across markets. The percentile series is aligned by date with the return data before analysis.

**Periods:** The analysis uses 1 January 2025 as the dividing date. Results are available for 2015–2024, 2025, 2026 year-to-date and since 2025. The 2026 period is partial and should not be treated as a full-year result.

**What it shows:**

- **Joint probability distribution:** A two-dimensional Gaussian kernel density estimate (KDE) is rendered as an interactive 3D surface. The X-axis is daily log return in percent, the Y-axis is trailing volume percentile from 0 to 100, and the Z-axis is estimated joint density. Higher areas indicate return-and-activity combinations that are more concentrated in the selected sample. This describes historical observations; it is not a forecast or evidence that volume causes a return.
- **Side-by-side comparison:** Select one asset and period, then optionally add a second asset or period. The 3D scenes share a density-axis scale, and accompanying 2D contour plots share a colour scale, making differences in shape and concentration easier to compare. Reflected KDE observations reduce estimation bias near the percentile boundaries of 0 and 100.
- **Risk measures:** A table reports observation count, mean daily return, annualised volatility, 95% Value at Risk (VaR), 95% Expected Shortfall, skewness, excess kurtosis and maximum drawdown. Annualised volatility is daily standard deviation multiplied by the square root of 252. VaR is the fifth-percentile daily loss; Expected Shortfall is the average loss among observations at or below that threshold. Maximum drawdown is calculated from the wealth path reconstructed from log returns.
- **Historical VaR backtest:** Each asset's 2015–2024 fifth-percentile loss threshold is held fixed and tested against returns since 2025. The table reports the number and rate of breaches against the nominal 5% rate. Kupiec's test evaluates whether the overall breach rate matches the expected rate; Christoffersen's test evaluates whether breaches are independent over time.
- **Rolling volatility:** A time-series chart displays annualised volatility using a selectable window of 21, 30, 60 or 90 trading days. It marks the start of 2025 to help identify when volatility changed relative to the analysis split.
- **Trading activity and return magnitude:** For each asset, the panel groups observations into low- and high-volume terciles within 2015–2024 and since 2025. It compares mean absolute daily log returns in the two groups, reports the high-minus-low gap, and calculates how that gap changed between periods. For Gold, it also compares its change with the average change for the available peer assets.
- **Distribution and variance tests:** Kolmogorov–Smirnov tests compare the pre-2025 and post-2024 return distributions, while Levene tests compare their variances. The table includes a post/pre volatility ratio, sample sizes and Holm-adjusted p-values to account for testing multiple selected assets.

**Controls and export:** Choose an asset and period for the main display, enable a second selection to compare distributions, and adjust KDE bandwidth from 0.65× to 1.4× to change the smoothing of the estimated surface. The panel labels the volume source used for each selected asset. A CSV download contains the selected assets' log returns and corresponding volume percentiles with clearly labelled columns.

**Interpreting the results:** A wider distribution along the return axis indicates more dispersed daily returns; movement toward negative returns indicates downside observations. In the activity table, a positive high-minus-low gap means mean absolute returns were larger in the high-volume group for that period. A positive Gold-versus-peer change means Gold's high-minus-low gap increased more—or decreased less—than the peer average. These comparisons are descriptive and are not causal estimates.

**Limitations:** ETF volume is a proxy for activity in a related market, not the exact volume of the futures or index whose returns are analysed. Continuous futures contracts can show volume disruptions around contract rolls. Early volume percentiles use fewer than 252 observations, and risk rows with fewer than 250 observations are flagged because tail estimates are less stable. The 2026 sample is incomplete. KS, Levene, Kupiec and Christoffersen p-values rely on assumptions that may not hold for clustered financial returns; the reported tests should therefore be treated as indicative rather than definitive. Historical distributions and backtests do not predict future performance.

## Contributions

| Member | Panel | Shared work |
|---|---|---|
| Charlie O'Sullivan | Regression analysis of silver, oil and gold | write this later |
| Kien Nguyen | Performance & ratios | Set up the repository and folder structure; built `data_loader.py` (price download, `get_prices()`, `get_returns()`) and later added the macro data (`get_macro()`, `get_macro_changes()`); built the overview grid in `app.py` and the dark theme; wrote the README apart from the other panel sections; kept `requirements.txt` up to date |
| Matthew Herbert | Correlation heatmap | Built Panel 3 (`panels/Matthew.py`): the significance-masked correlation heatmap, the rolling correlation graph with confidence bands and event shading, the year-by-year animated timeline, and the Decoupling Analysis with Fisher z-tests and the ranked change table. Wrote README for Panel 3. Assisted in layout of dashboard. Troubleshoot email relating to commits and later push/pull problems. Debugging. |
| Michael Gilgunn | Distribution & risk | write this later|

## Limitations
- Gold, silver, oil and copper are future prices, which jump slightly when contracts roll over.
- The S&P 500 is a price index, so it does not include dividends.
- The saved data stops on 30 October 2026. Run the refresh commands above to update it.
- The "since 2025" period is less than two years long, so its results are less reliable than the 2015–2024 ones.
- The dashboard shows how gold moves with other markets, not what causes those moves.
- This project was built for a university module and is not investment advice.