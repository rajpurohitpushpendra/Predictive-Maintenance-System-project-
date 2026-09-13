# Predictive Maintenance System

A machine learning based predictive maintenance system that predicts whether an industrial machine is likely to fail using sensor parameters such as temperature, rotational speed, torque, and tool wear.

The project includes data analysis, feature engineering, machine learning model training, model evaluation, and an interactive Streamlit dashboard for machine failure prediction.

## Features

- Data cleaning and preprocessing
- Exploratory Data Analysis
- Feature engineering
- Machine failure prediction
- Random Forest classification
- XGBoost classification
- Model evaluation using Accuracy, Precision, Recall, and F1 Score
- Interactive Streamlit dashboard
- Failure probability prediction

## Dataset

This project uses the AI4I 2020 Predictive Maintenance Dataset.

The dataset contains 10,000 machine records with industrial sensor parameters and machine failure information.

Selected features used in this project:

- Air Temperature
- Process Temperature
- Rotational Speed
- Torque
- Tool Wear

The target variable is Machine Failure.

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- Matplotlib
- Seaborn
- Streamlit
- Joblib
- Jupyter Notebook

## Machine Learning Models

Two classification models were trained and compared:

### Random Forest

Random Forest was trained with balanced class weights to handle the imbalance between failure and non-failure cases.

### XGBoost

XGBoost was trained as a second classification model and achieved better overall performance than Random Forest on the test dataset.

## Feature Engineering

A new feature called `Temperature difference` was created:

```text
Temperature difference = Process Temperature - Air Temperature

This feature represents the difference between the machine's process temperature and air temperature.

Model Performance
Model	Accuracy	Precision	Recall	F1 Score
Random Forest	98.25%	74.63%	73.53%	74.07%
XGBoost	98.75%	87.72%	73.53%	80.00%

XGBoost achieved the highest accuracy and F1 Score among the two models.

Streamlit Dashboard

The project includes an interactive Streamlit dashboard where users can enter machine sensor parameters and receive a machine failure prediction.

Users can provide:

Air Temperature
Process Temperature
Rotational Speed
Torque
Tool Wear

The dashboard also displays the predicted failure probability and model performance.

Project Structure
PMS_project/
│
├── data/
│   └── predictive_maintenance.csv
│
├── models/
│   ├── random_forest.pkl
│   └── xgboost.pkl
│
├── notebooks/
│   └── analysis.ipynb
│
├── screenshots/
│   └── dashboard.png
│
├── app.py
├── data_check.py
├── eda.py
├── train_model.py
├── requirements.txt
├── README.md
└── .gitignore
Installation

Clone the repository:

git clone https://github.com/rajpurohitpushpendra/Predictive-Maintenance-System-project-.git
cd Predictive-Maintenance-System-project-

Create a virtual environment:

python3 -m venv venv

Activate the virtual environment:

source venv/bin/activate

Install the required libraries:

pip install -r requirements.txt
Run the Application

Start the Streamlit dashboard:

streamlit run app.py
Model Training

To retrain the machine learning models:

python train_model.py

The trained models will be saved inside the models directory.

Results

The system successfully predicts machine failure using industrial sensor parameters.

XGBoost performed better than Random Forest based on the test results, achieving an accuracy of 98.75% and an F1 Score of 80.00%.

Future Improvements
Add more machine sensor parameters
Add maintenance recommendations
Add prediction history
Add interactive data visualizations
Deploy the Streamlit application online
Improve failure detection using additional machine learning techniques
Disclaimer

This project is developed for educational and demonstration purposes using the AI4I 2020 Predictive Maintenance Dataset. The dataset is not actual industrial or ONGC production data.