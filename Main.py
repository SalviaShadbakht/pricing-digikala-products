from src.Preprocess.data_preprocessing import DataPreprocessor
from src.Model.model import CategoryModelTrainer
import pandas as pd
import argparse

def main(args):
    if (args.hasDataPreprocessed):
        preprocessed_data = pd.read_csv('Data/data_preprocessed.csv', low_memory=False)
    else:
        # Define your file_path
        file_path = args.dataPath

        # Instantiate the DataPreprocessor class
        preprocessor = DataPreprocessor(file_path)

        # Perform preprocessing
        preprocessed_data = preprocessor.preprocess_data()

        preprocessed_data.to_csv('Data/data_preprocessed.csv', index=False)


    # Continue with modeling
    trainer = CategoryModelTrainer(preprocessed_data)
    
    if(args.model=='xgboost'):
        trainer.train_models_XGBoost()
    elif(args.model=='randomforest'):
        trainer.train_models_randomforest()
    elif(args.model=='svr'):
        trainer.train_models_SVR()
    elif(args.model=='ann'):
        trainer.train_models_ANN()
    else:
        print("Choose one model for traing: xgboost, randomforest, svr, ann")



if __name__ == "__main__":

    parser = argparse.ArgumentParser('Digikala Task parameters')
    parser.add_argument('--model', type=str, default='xgboost',
                    help='Which model to train')
    parser.add_argument('--dataPath', type=str, default='Data/train.csv',
                    help='dataPath')
    parser.add_argument('--hasDataPreprocessed', type=bool, default=False,
                    help='dataPath')
    
    args = parser.parse_args()

    main(args)

