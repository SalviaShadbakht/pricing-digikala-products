import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

class CategoryModelTrainer:
    def __init__(self, df):
        self.df = df
        self.label_encoder = LabelEncoder()
        self.models = {}
        self.predictions = {}
        self.y_test_categories_list = {}
        self.mape_per_category = {}
        self.kf = KFold(n_splits=5, shuffle=True, random_state=42)

    def train_models_randomforest(self):
        for idx, category in enumerate(self.df['دسته بندی'].unique()):
            df_category = self.df[self.df['دسته بندی'] == category].copy()
            df_category.dropna(axis=1, how='all', inplace=True)
            non_price_columns = df_category.columns[df_category.columns != 'price']
            df_category[non_price_columns] = df_category[non_price_columns].astype(str)

            for column in non_price_columns:
                df_category[column] = self.label_encoder.fit_transform(df_category[column])

            X_category = df_category.drop(['price', 'دسته بندی'], axis=1)
            y_category = df_category['price']

            mape_per_fold = []
            model_per_fold = []
            prediction_per_fold = []
            y_test_per_fold = []

            for fold_idx, (train_index, test_index) in enumerate(self.kf.split(X_category)):
                X_train_category, X_test_category = X_category.iloc[train_index], X_category.iloc[test_index]
                y_train_category, y_test_category = y_category.iloc[train_index], y_category.iloc[test_index]

                model_category = RandomForestRegressor(n_estimators=100, random_state=42)
                model_category.fit(X_train_category, y_train_category)

                predictions_category_fold = model_category.predict(X_test_category)

                mape_fold = mean_absolute_percentage_error(y_category.iloc[test_index], predictions_category_fold)

                mape_per_fold.append(mape_fold)
                model_per_fold.append(model_category)
                prediction_per_fold.append(predictions_category_fold)
                y_test_per_fold.append(y_category.iloc[test_index])

            best_fold_idx = np.argmin(mape_per_fold)

            self.predictions[category] = prediction_per_fold[best_fold_idx]
            self.y_test_categories_list[category] = y_test_per_fold[best_fold_idx]
            self.models[category] = model_per_fold[best_fold_idx]

            print(f"MAPE Category {category}: Mean= {np.mean(mape_per_fold):.2f}%, Variance={np.var(mape_per_fold):.2f}%")

        selected_categories = list(self.models.keys())
        selected_predictions = np.concatenate([self.predictions[category] for category in selected_categories])
        selected_actuals = np.concatenate([self.y_test_categories_list[category] for category in selected_categories])

        mape_selected = np.mean(np.abs((selected_actuals - selected_predictions) / selected_actuals)) * 100
        print(f'\n\nFinal Mean Absolute Percentage Error: {mape_selected:.2f}%')

    def preprocess_category(self, df_category):
        df_category.dropna(axis=1, how='all', inplace=True)
        non_price_columns = df_category.columns[df_category.columns != 'price']
        df_category[non_price_columns] = df_category[non_price_columns].astype(str)

        for column in non_price_columns:
            df_category[column] = self.label_encoder.fit_transform(df_category[column])

        X_category = df_category.drop(['price', 'دسته بندی'], axis=1)
        y_category = df_category['price']

        # Standardize the features and target using StandardScaler
        feature_scaler = StandardScaler()
        target_scaler = StandardScaler()

        # Standardize the features and target
        X_category_scaled = feature_scaler.fit_transform(X_category)
        y_category_scaled = target_scaler.fit_transform(y_category.values.reshape(-1, 1)).flatten()

        return X_category_scaled, y_category_scaled, target_scaler

    def train_models_SVR(self):
        for idx, category in enumerate(self.df['دسته بندی'].unique()):
            df_category = self.df[self.df['دسته بندی'] == category].copy()
            X_category_scaled, y_category_scaled,target_scaler = self.preprocess_category(df_category)

            mape_per_fold = []
            model_per_fold = []
            prediction_per_fold = []
            y_test_per_fold = []

            for fold_idx, (train_index, test_index) in enumerate(self.kf.split(X_category_scaled)):
                X_train_category, X_test_category = X_category_scaled[train_index], X_category_scaled[test_index]
                y_train_category, y_test_category = y_category_scaled[train_index], y_category_scaled[test_index]

                model_category = SVR(kernel='linear')  # You can choose different kernels such as 'linear', 'rbf', etc.
                model_category.fit(X_train_category, y_train_category)

                predictions_category_fold = model_category.predict(X_test_category)
                predictions_category_fold = target_scaler.inverse_transform(predictions_category_fold.reshape(-1, 1)).flatten()

                mape_fold = mean_absolute_percentage_error(df_category['price'].iloc[test_index], predictions_category_fold)

                mape_per_fold.append(mape_fold)
                model_per_fold.append(model_category)
                prediction_per_fold.append(predictions_category_fold)
                y_test_per_fold.append(df_category['price'].iloc[test_index])

            best_fold_idx = np.argmin(mape_per_fold)

            self.predictions[category] = prediction_per_fold[best_fold_idx]
            self.y_test_categories_list[category] = y_test_per_fold[best_fold_idx]
            self.models[category] = model_per_fold[best_fold_idx]

            print(f"MAPE Category {category}: Mean= {np.mean(mape_per_fold):.2f}%, Variance={np.var(mape_per_fold):.2f}%")

        selected_categories = list(self.models.keys())
        selected_predictions = np.concatenate([self.predictions[category] for category in selected_categories])
        selected_actuals = np.concatenate([self.y_test_categories_list[category] for category in selected_categories])

        mape_selected = np.mean(np.abs((selected_actuals - selected_predictions) / selected_actuals)) * 100
        print(f'\n\nFinal Mean Absolute Percentage Error: {mape_selected:.2f}%')

    def train_models_XGBoost(self):
            for idx, category in enumerate(self.df['دسته بندی'].unique()):
                df_category = self.df[self.df['دسته بندی'] == category].copy()
                X_category_scaled, y_category_scaled,target_scaler = self.preprocess_category(df_category)

                mape_per_fold = []
                model_per_fold = []
                prediction_per_fold = []
                y_test_per_fold = []

                for fold_idx, (train_index, test_index) in enumerate(self.kf.split(X_category_scaled)):
                    X_train_category, X_test_category = X_category_scaled[train_index], X_category_scaled[test_index]
                    y_train_category, y_test_category = y_category_scaled[train_index], y_category_scaled[test_index]

                    model_category = XGBRegressor(n_estimators=100, random_state=42)
                    model_category.fit(X_train_category, y_train_category)
                    
                    predictions_category_fold = model_category.predict(X_test_category)
                    predictions_category_fold = target_scaler.inverse_transform(predictions_category_fold.reshape(-1, 1)).flatten()

                    mape_fold = mean_absolute_percentage_error(df_category['price'].iloc[test_index], predictions_category_fold)

                    mape_per_fold.append(mape_fold)
                    model_per_fold.append(model_category)
                    prediction_per_fold.append(predictions_category_fold)
                    y_test_per_fold.append(df_category['price'].iloc[test_index])

                best_fold_idx = np.argmin(mape_per_fold)

                self.predictions[category] = prediction_per_fold[best_fold_idx]
                self.y_test_categories_list[category] = y_test_per_fold[best_fold_idx]
                self.models[category] = model_per_fold[best_fold_idx]

                print(f"MAPE Category {category}: Mean= {np.mean(mape_per_fold):.2f}%, Variance={np.var(mape_per_fold):.2f}%")

            selected_categories = list(self.models.keys())
            selected_predictions = np.concatenate([self.predictions[category] for category in selected_categories])
            selected_actuals = np.concatenate([self.y_test_categories_list[category] for category in selected_categories])

            mape_selected = np.mean(np.abs((selected_actuals - selected_predictions) / selected_actuals)) * 100
            print(f'\n\nFinal Mean Absolute Percentage Error: {mape_selected:.2f}%')
    

    def build_ann_model(self, input_dim):
        model = Sequential()
        model.add(Dense(64, input_dim=input_dim, activation='relu'))
        model.add(Dense(1, activation='linear'))
        model.compile(loss='mean_squared_error', optimizer='adam')
        return model
    
    def train_models_ANN(self):
            for idx, category in enumerate(self.df['دسته بندی'].unique()):
                df_category = self.df[self.df['دسته بندی'] == category].copy()
                X_category_scaled, y_category_scaled,target_scaler = self.preprocess_category(df_category)

                mape_per_fold = []
                model_per_fold = []
                prediction_per_fold = []
                y_test_per_fold = []

                for fold_idx, (train_index, test_index) in enumerate(self.kf.split(X_category_scaled)):
                    X_train_category, X_test_category = X_category_scaled[train_index], X_category_scaled[test_index]
                    y_train_category, y_test_category = y_category_scaled[train_index], y_category_scaled[test_index]

                    model_category = self.build_ann_model(X_train_category.shape[1])
                    model_category.fit(X_train_category, y_train_category, epochs=5, batch_size=32, verbose=0)

                    
                    predictions_category_fold = model_category.predict(X_test_category)
                    predictions_category_fold = target_scaler.inverse_transform(predictions_category_fold.reshape(-1, 1)).flatten()

                    mape_fold = mean_absolute_percentage_error(df_category['price'].iloc[test_index], predictions_category_fold)

                    mape_per_fold.append(mape_fold)
                    model_per_fold.append(model_category)
                    prediction_per_fold.append(predictions_category_fold)
                    y_test_per_fold.append(df_category['price'].iloc[test_index])

                best_fold_idx = np.argmin(mape_per_fold)

                self.predictions[category] = prediction_per_fold[best_fold_idx]
                self.y_test_categories_list[category] = y_test_per_fold[best_fold_idx]
                self.models[category] = model_per_fold[best_fold_idx]

                print(f"MAPE Category {category}: Mean= {np.mean(mape_per_fold):.2f}%, Variance={np.var(mape_per_fold):.2f}%")

            selected_categories = list(self.models.keys())
            selected_predictions = np.concatenate([self.predictions[category] for category in selected_categories])
            selected_actuals = np.concatenate([self.y_test_categories_list[category] for category in selected_categories])

            mape_selected = np.mean(np.abs((selected_actuals - selected_predictions) / selected_actuals)) * 100
            print(f'\n\nFinal Mean Absolute Percentage Error: {mape_selected:.2f}%')








