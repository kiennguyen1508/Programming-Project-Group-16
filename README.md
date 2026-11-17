# Gold and its Drivers: Economic & Finance Dashboard

This interactive dashboard explores how the price of gold is related to, and affected by, other markets and economic factors between 2015 and 2026. It compares gold with silver, oil, copper and the S&P 500, and uses the VIX to capture market fear. Users can choose the time period, assets and settings in each panel to compare how these relationships changed over time.

## Quick start
Tested with Python 3.14.7 on macOS.

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
streamlit run app.py
```
The dashboard opens in your browser at http://localhost:8501.
If that port is busy, Streamlit uses the next one (8502) and prints the address in the terminal.

No internet connection or API key is needed to run the app. The data is already saved in the `data/` folder.