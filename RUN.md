# Run locally

1. Create and activate a virtual environment (Python 3.11 or newer):
```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
```

2. Install the dependencies:
```bash
python -m pip install -r requirements-train.txt
```

3. Download the dataset [Dataset](https://www.kaggle.com/datasets/ivantha/sri-lanka-property-ads-dataset?select=properties.csv) in to data folder. 

4. Run `sample_data.py` script in *src* folder to sample the dataset.
```bash
python src/sample_data.py
```

5. Train the models (saves the best model to `models/`):
```bash
python -m src.train
```
For a faster run, set fewer tuning trials:
```bash
N_TRIALS=10 NGB_TRIALS=5 python -m src.train                          # macOS/Linux
$env:N_TRIALS="10"; $env:NGB_TRIALS="5"; python -m src.train          # Windows PowerShell
```

6. Run the tests:
```bash
python -m pytest
```

7. Start the app:
```bash
streamlit run app.py
```
The app opens in your browser at http://localhost:8501.

To explore the full analysis (EDA, SHAP, LIME), open `notebooks/house_price_prediction.ipynb` and run all cells.
