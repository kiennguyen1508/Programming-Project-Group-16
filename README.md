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

###  Panel 1: Charlie's panel


###  Panel 2: Performance and ratios (Kien Nguyen)
**Question:** How has gold performed against the other assets, and does it actually protect investors when markets are stressed?

**Data:** Gold, silver, Brent oil and the S&P 500 from `prices.csv`, plus the VIX and copper from `macro.csv`.

**What it shows:** The panel has three tabs.
- **Performance:** how each asset grew over the chosen period, or how far it fell from its previous peak (drawdown). A table compares annual return, volatility, Sharpe ratio and Sortino ratio (both using an assumed 4% yearly risk-free rate) and maximum drawdown, with the best value in each column highlighted.
- **Stress days:** each asset's daily return on calm days and on stressed days. A day counts as stressed if the VIX closed at or above the chosen level the day before.
- **Ratios:** the gold/silver, gold/oil and gold/copper ratios, and how often each one moved back towards its average after drifting far from it. The average is a rolling 5-year one, so the test only uses data that was available at the time.

**Controls:** period, assets, a start-date slider, a Growth/Drawdown switch and a Replay button that animates the chart. The Stress days tab also has a VIX threshold (25 by default) and a choice between the average and the median.

**Key findings:** Gold did not reliably rise after fear spikes. Over the full period it averaged +0.01% a day on the 391 stressed days, compared with +0.05% on calm days, while the S&P 500 averaged +0.13% on stressed days. The gold/silver ratio was the most predictable of the three: when it was more than 2σ away from its rolling 5-year average, it moved back towards that average over the next 3 months 81% of the time, against a base rate of about 43-57%.

###  Panel 3: Correlation Heatmap (Matthew Herbert)
**Question:** Which assets move together with gold, and did those links change in relation to various economic events?

**Data:** My panel draws on four files in the `data` folder:
- `data/prices.csv`: This is the daily closing prices for Gold, Silver, Oil and the S&P 500, from Yahoo Finance, which is called by the `get_returns()` function in `data_loader.py` python file.
- `data/macro.csv` : This is the daily levels of Commodities, the Dollar and Copper also from Yahoo Finance, the 10-year yield and the VIX, called by `load_macro()` in my panel.
- `data/DFII10.csv`: This file supplies the real yield column figures in `macro.csv`.
- `data/VIX_History.csv` : This supplies the VIX column figures in `macro.csv`. 

Gold, Silver, Oil and the S&P 500 use daily log returns. The other assets use the daily percentage change, apart from the real yield, which uses the daily change in percentage points. In certain trading days, there is not complete data on each of the analysed and used assets. In these cases, the trading day as a whole is excluded, leaving apporximately 
2,900 complete trading days from Jan 2015, to October 2026, which my panel uses.

**What Panel 3 Shows:**
- ***Correlation Heatmap:*** A staircase heatmap of the Pearson correlation (r) for every pair of chosen assets. Gold means the pair moves together and blue means they move in opposite directions. Pairs that are not significant at the chosen level are greyed out and marked "n.s.", so the eye goes only to the links that are real.
- ***Rolling Graph:*** How the correlation between two assets changes over time, with a shaded 95% confidence band. Shaded boxes mark the COVID crash, the Ukraine war and rate hikes, and the selected period. A dotted line marks the silver peak (26 Jan 2026). Dotted average lines compare before and during the selected period, and a row of cards shows the latest, average, lowest and highest r. A second pair can be added for comparison.
- ***Animated Timeline:*** One heatmap per year from 2015 to 2026, played with a Play button or a year slider, to show how the links shift year by year. 2026 covers January to October only.
- ***Decoupling Analysis:*** Compares the selected period with all the days before it. A table ranks every pair by how much its correlation changed, with a Fisher z-test p-value for each change, and can be downloaded as a CSV. It can also be shown as a heatmap of the changes. The "Pair in Focus" section then tests one chosen pair and says in plain words whether the change is real or just random noise.
###  Panel 4: Michael's panel

## Contributions

| Member | Panel | Shared work |
|---|---|---|
| Charlie | Silver vs gold regression | write this later |
| Kien Nguyen | Performance & ratios | Set up the repository and folder structure; built `data_loader.py` (price download, `get_prices()`, `get_returns()`) and later added the macro data (`get_macro()`, `get_macro_changes()`); built the overview grid in `app.py` and the dark theme; wrote the README apart from the other panel sections; kept `requirements.txt` up to date |
| Matthew Herbert | Correlation heatmap | write this later |
| Michael Gilgunn | Distribution & risk | write this later|

## Limitations
- Gold, silver, oil and copper are future prices, which jump slightly when contracts roll over.
- The S&P 500 is a price index, so it does not include dividends.
- The saved data stops on 30 October 2026. Run the refresh commands above to update it.
- The "since 2025" period is less than two years long, so its results are less reliable than the 2015–2024 ones.
- The dashboard shows how gold moves with other markets, not what causes those moves.
- This project was built for a university module and is not investment advice.