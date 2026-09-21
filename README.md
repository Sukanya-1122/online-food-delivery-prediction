# Online Food Delivery Prediction

This project trains a binary classification model that predicts the `Output` column:

- `Yes`: the customer is predicted to use/order online food delivery.
- `No`: the customer is predicted not to use/order online food delivery.

The project uses the verified feature columns from the uploaded dataset. The unnecessary `Unnamed: 13` column is removed during loading and is never used as a feature.

## Project structure

```text
online-food-delivery-prediction/
├── data/
│   └── online food delivery dataset.csv
├── models/
│   └── food_delivery_model.pkl
├── train_model.py
├── app.py
├── requirements.txt
└── README.md
```

## Verified dataset details

The source dataset contains 388 rows and 14 columns, including `Unnamed: 13`. It has 103 duplicate rows, no missing values in the uploaded version, and these target values:

- `Yes`: 301 rows
- `No`: 87 rows

Training removes duplicate rows and the `Unnamed: 13` index-like column. Pipeline imputers are still included so the saved preprocessing remains robust to missing values.

## Features

The model uses exactly these 12 input features:

`Age`, `Gender`, `Marital Status`, `Occupation`, `Monthly Income`, `Educational Qualifications`, `Family size`, `Customer Type`, `latitude`, `longitude`, `Pin code`, and `Feedback`.

Numeric columns are median-imputed and standardized. Categorical columns are mode-imputed and one-hot encoded with `handle_unknown="ignore"`. All preprocessing is fitted inside each model pipeline using only the training split.

## Install and train

From the `online-food-delivery-prediction` directory:

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python train_model.py
```

macOS/Linux:

```bash
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python train_model.py
```

The training command compares Logistic Regression, Decision Tree, and Random Forest using accuracy, precision, recall, F1-score, confusion matrices, and classification reports. The final model is selected using overall positive-class performance, prioritizing F1-score, then recall, precision, and accuracy. The complete model and preprocessing pipeline are saved to `models/food_delivery_model.pkl`.

## Run the Streamlit application

```bash
streamlit run app.py
```

Open the local URL displayed by Streamlit, normally `http://localhost:8501`.

## GitHub upload commands

Run these commands from the project directory after creating an empty GitHub repository:

```bash
git init
git add .
git commit -m "Build online food delivery prediction app"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/online-food-delivery-prediction.git
git push -u origin main
```

Do not commit secrets or a virtual environment. Add `.venv/` and `__pycache__/` to a `.gitignore` file if needed.

## Deployment

### Streamlit Community Cloud

1. Push the project to GitHub.
2. Create a new app at `share.streamlit.io`.
3. Select the repository, `main` branch, and `app.py` as the main file.
4. Confirm that `requirements.txt`, the `data` folder, and the trained `models/food_delivery_model.pkl` file are committed.
5. Deploy the app.

### Render

1. Push the project to GitHub.
2. Create a new Render Web Service connected to the repository.
3. Use this build command:

```bash
pip install -r requirements.txt
```

4. Use this start command:

```bash
streamlit run app.py --server.address 0.0.0.0 --server.port $PORT
```

5. Use Python 3.10 or newer and deploy. The model artifact must be present in `models/` before deployment.
