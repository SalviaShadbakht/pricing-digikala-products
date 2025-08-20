# Digikala Product Pricing Prediction

## Project Overview
This project predicts product prices on Digikala using machine learning models. The pipeline includes data preprocessing, feature engineering, exploratory data analysis (EDA), and model training with multiple algorithms. The goal is to build accurate predictive models and evaluate their performance using MAPE (Mean Absolute Percentage Error).

---

## Features
- Clean and preprocess Digikala product pricing data.  
- Perform Exploratory Data Analysis (EDA) to understand data patterns.  
- Implement multiple machine learning models:  
  - Random Forest  
  - Support Vector Regression (SVR)  
  - XGBoost  
  - Simple Artificial Neural Network (ANN)  
- Evaluate models using MAPE to measure prediction accuracy.

---

## Project Structure
SalviaShadbakht-Digikala_Task/
│
├── Main.py # Main script to run the project
├── requirements.txt # Packages required for the project
├── Data/ # Dataset files
├── src/ # Modules: preprocess and model
└── Notebooks/ # Jupyter notebooks for EDA and experiments

## Installation
1. Clone the repository:
git clone git@github.com:SalviaShadbakht/News-Editor-RAG.git

2. Navigate to the project directory:
cd SalviaShadbakht-Digikala_Task

4. Create a virtual environment and install dependencies:
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate
pip install -r requirements.txt


## Usage

Run the project using:
python Main.py --model your_model --dataPath your_data_path --hasDataPreprocessed True

Arguments:

--model : Choose the model ('ann', 'xgboost', 'svr', 'randomforest')
--dataPath : Path to your dataset
--hasDataPreprocessed : Set True if data is already preprocessed

Default values are set, so you can also run the script without any arguments.


## Results

Models are evaluated using MAPE.
XGBoost achieved approximately 80% MAPE, indicating room for improvement in data preprocessing and model optimization.
Detailed analysis and experiments are included in the Notebooks folder.



