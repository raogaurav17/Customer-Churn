# Customer Churn Prediction

This project analyzes customer data to predict churn using machine learning models.

## Project Structure

- `app.py` - Main application file (for deployment or API, if applicable)
- `customer_churn_data.csv` - Dataset used for training and testing
- `dataExploration.ipynb` - Jupyter notebook for data exploration and visualization
- `PredictiveAnalysis.ipynb` - Jupyter notebook for model training, evaluation, and prediction
- `model.pkl` - Saved machine learning model
- `scaler.pkl` - Saved scaler for feature normalization
- `.ipynb_checkpoints/` - Jupyter notebook checkpoints

## Getting Started

1. **Clone the repository**
2. **Install dependencies**

   ```sh
   pip install -r requirements.txt
   ```

3. **Run notebooks**

   - Open `dataExploration.ipynb` for EDA.
   - Open `PredictiveAnalysis.ipynb` for model training and evaluation.

4. **Run the app**
   ```sh
   ppython -m streamlit run app.py
   ```

## Notebooks Overview

- **dataExploration.ipynb**: Data cleaning, visualization, and feature engineering.
- **PredictiveAnalysis.ipynb**: Model selection, training, evaluation (classification report, confusion matrix), and saving models.

## Model Artifacts

- `model.pkl`: Trained model for prediction.
- `scaler.pkl`: Scaler used for feature normalization.

## Ignore Files

The following files are ignored via `.gitignore`:

- Notebook checkpoints
- Model and scaler artifacts
- Data files
- Python cache and environment files

---

_For questions or contributions, please open an issue or submit a pull request._
