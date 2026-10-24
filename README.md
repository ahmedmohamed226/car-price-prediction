# Car Price Prediction

A small ML project that predicts the selling price of a used car based on things like its age, how many km it was driven, fuel type, transmission etc.

The data is from Kaggle (CarDekho used cars): https://www.kaggle.com/datasets/nehalbirla/vehicle-dataset-from-cardekho
It gets downloaded automatically with kagglehub when you run the notebook.

## What I did

I extracted the car brand from the name column and made an age column from the year, since age usually matters more than the year itself. Then I encoded the categorical columns (fuel type, seller type, transmission, owner, brand) with LabelEncoder and trained 3 models: LinearRegression, RandomForest and GradientBoosting, comparing them with R2, MAE and RMSE.

RandomForest usually ends up being the best one on this dataset.

## Project structure

```
car-price-prediction/
├── notebooks/
│   └── car_price_model.ipynb   # training notebook
├── app/
│   └── streamlit_app.py        # small web app to try the model
├── requirements.txt
└── .gitignore
```

## How to run

1. Open `notebooks/car_price_model.ipynb` in Google Colab and run all cells. This downloads the dataset, trains the models and saves the best one in a `model_artifacts` folder.
2. Download the `model_artifacts` folder from Colab and put it in the project root.
3. Install the requirements and run the app:

```bash
pip install -r requirements.txt
streamlit run app/streamlit_app.py
```

The `model_artifacts` folder is not pushed to GitHub because of its size, you generate it locally by running the notebook.

## Libraries used

pandas, numpy, scikit-learn, matplotlib, seaborn, joblib, kagglehub, streamlit
