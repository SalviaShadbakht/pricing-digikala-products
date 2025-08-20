import pandas as pd
import numpy as np
import re
from itertools import combinations
from fuzzywuzzy import fuzz
import ast
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
import os

class DataPreprocessor:
    def __init__(self, file_path):
        self.file_path = file_path
        self.data = None
        self.columns = set()  # Initialize the 'columns' attribute

    def load_data(self):
        # Load data from file_path
        self.data = pd.read_csv(self.file_path)

    # @staticmethod
    def preprocess_product_description(self,description):
        characters_to_remove = ['\r', '\n', '\t']

        def remove_characters(s):
            for char in characters_to_remove:
                s = s.replace(char, ' ')
            return s

        def clean_value(value):
            if isinstance(value, list):
                return [remove_characters(item) if isinstance(item, str) else item for item in value]
            else:
                return remove_characters(value)

        def replace_special_sequences(s):
            if isinstance(s, str):
                # Replace 'w\\/' with 'with'
                s = s.replace('w\\/', 'with')
                # Replace '\\/' with '/'
                s = s.replace('\\/', '/')
                # Replace '\\' with '/'
                s = s.replace('\\\\', '\/')
            return s

        def remove_u_x(value):
            # Remove \u20 followed by four characters
            value = re.sub(r'\\u\w\w\w\w', '', value)
            # Remove \x0 followed by two characters
            value = re.sub(r'\\x\w\w', '', value)
            return value

        dictionary_extracted = ast.literal_eval(description)
        cleaned_dict = {remove_characters(key): clean_value(value) for key, value in dictionary_extracted.items()}
        cleaned_dict = {replace_special_sequences(key): replace_special_sequences(value) for key, value in cleaned_dict.items()}
        cleaned_description = str(cleaned_dict)

        # Apply remove_u_x to the cleaned description
        cleaned_description = remove_u_x(cleaned_description)

        return cleaned_description
    
    @staticmethod
    def make_separate(x, columns_set):
        dictionary_extracted = ast.literal_eval(x)
        for item in dictionary_extracted.keys():
            columns_set.add(item)
        return x

    @staticmethod
    def add_data(x, columns_set):
        dictionary_extracted = ast.literal_eval(x["product_description"])
        for item in dictionary_extracted.keys():
            x[item] = dictionary_extracted[item]
        return x

    def make_dataSet(self):
        # Apply the 'make_separate' function to populate the 'columns' set
        self.data["product_description"].map(lambda x: self.make_separate(x, self.columns))

        # Create a new DataFrame with columns extracted from 'test' function
        new_df_columns = pd.DataFrame(columns=list(self.columns))
        self.data = pd.concat([self.data, new_df_columns], axis=1)

        # Apply the 'add_data' function to add data to new columns
        self.data = self.data.apply(lambda x: self.add_data(x, self.columns), axis=1)

    
    # Find Category with few data or can be merged
    def find_and_delete_few_data(self):
        category_column='دسته بندی'
        grouped_data = self.data.groupby(category_column).size().reset_index(name='row_count')
        grouped_data.sort_values(by='row_count', ascending=True)
        categories_to_remove = grouped_data[grouped_data['row_count'] < 20][category_column].tolist()
        self.data = self.data[~self.data[category_column].isin(categories_to_remove)]
    
    def find_and_delete_similar_column(self):
        category_column = 'دسته بندی'  # Replace with the actual column name

        # Create a list to store the common columns for each category
        common_columns_by_category = {}

        # Iterate over each category
        for category in self.data[category_column].unique():
            # Filter the DataFrame for the current category
            category_df = self.data[self.data[category_column] == category]

            # Identify columns with non-null values for the current category
            non_null_columns = category_df.columns[category_df.notnull().any()].tolist()

            # Store the non-null columns in the dictionary
            common_columns_by_category[category] = non_null_columns

        # Initialize a dictionary to store duplicate categories and their indices
        duplicate_categories = {}

        # Loop through each category and its non-null columns
        for category1, columns1 in common_columns_by_category.items():
            for category2, columns2 in common_columns_by_category.items():
                if category1 != category2 and set(columns1) == set(columns2):
                    # Found two categories with the same non-null columns
                    if category1 not in duplicate_categories:
                        duplicate_categories[category1] = {"categories": [category2], "columns": columns1}
                    else:
                        duplicate_categories[category1]["categories"].append(category2)

        for category1, data1 in duplicate_categories.items():
            for category2 in data1["categories"]:
                # Get the indices of rows to update
                indices_to_update = self.data[self.data[category_column].isin([category1, category2])].index

                # Create a new concatenated list of categories
                concatenated_categories = sorted(set([category1] + data1['categories']))

                # Set the new concatenated list of categories for the selected rows
                self.data.loc[indices_to_update, category_column] = ', '.join(concatenated_categories)

    def find_and_merge_category_with_similar_columns_and_price(self):
        category_column = 'دسته بندی' 

        # Group by category and get the column indices with at least one non-null value
        non_null_columns_indices_by_category = self.data.groupby(category_column).apply(lambda x: np.nonzero(x.notnull().any().values)[0].tolist())

        # Calculate the threshold for common indices
        common_indices_threshold = 0.8

        # List to store pairs of categories with common indices
        common_indices_categories = []

        # Iterate through each pair of categories and find those with at least 80% common indices
        for i, (category1, indices1) in enumerate(non_null_columns_indices_by_category.items()):
            for category2, indices2 in list(non_null_columns_indices_by_category.items())[i+1:]:
                common_indices = set(indices1) & set(indices2)
                common_indices_percentage = len(common_indices) / max(len(indices1), len(indices2))
                
                if common_indices_percentage >= common_indices_threshold:
                    common_indices_categories.append((category1, category2, common_indices_percentage))

        # Initialize a dictionary to store categories and their corresponding common indices
        category_indices_dict = {}

        # Populate the dictionary
        for category1, category2, common_indices_percentage in common_indices_categories:
            if category1 not in category_indices_dict:
                category_indices_dict[category1] = set()
            if category2 not in category_indices_dict:
                category_indices_dict[category2] = set()

            category_indices_dict[category1].add(category2)
            category_indices_dict[category2].add(category1)

        # Function to find groups of categories with at least 80% common indices
        def find_category_groups(category_indices_dict, common_indices_threshold):
            visited = set()
            groups = []

            def dfs(category, group):
                if category not in visited:
                    visited.add(category)
                    group.add(category)

                    for neighbor in category_indices_dict[category]:
                        if neighbor not in visited:
                            dfs(neighbor, group)

            for category in category_indices_dict:
                if category not in visited:
                    group = set()
                    dfs(category, group)
                    groups.append(group)

            return groups

        # Find groups of categories with at least 80% common indices
        common_indices_threshold = 0.8
        category_groups = find_category_groups(category_indices_dict, common_indices_threshold)

        for i, group in enumerate(category_groups, start=1):
            concatenated_categories = "_".join(sorted(self.data[self.data[category_column].isin(group)][category_column].unique()))
            self.data.loc[self.data[category_column].isin(group), category_column] = concatenated_categories

        completely_null_columns = self.data.columns[self.data.isnull().all()]
        self.data = self.data.drop(columns=completely_null_columns)

    def delete_columns_less_5data(self):
        columns_under_5_samples = []
        for column in self.data.columns:
            non_null_count = self.data[column].count()
            if non_null_count < 5:
                columns_under_5_samples.append(column)
        self.data = self.data.drop(columns=columns_under_5_samples)

    def find_similar_category_name_to_merge(self):
        # Set a similarity threshold for identifying similar column names
        similarity_threshold = 90

        # Get all combinations of column names
        column_combinations = list(combinations(self.data.columns, 2))

        # Find similar column name pairs
        similar_columns = [(col1, col2) for col1, col2 in column_combinations if fuzz.ratio(col1, col2) > similarity_threshold]

        def group_similar_columns(similar_columns, data):
            grouped_columns = []

            for col1, col2 in similar_columns:
                # Find rows where at least one of the two columns has values
                mask = data[col1].notnull() & data[col2].notnull()

                # Check if there are at most 5 rows where both columns have values
                if 0 <= sum(mask) <= 3:
                    # Check if the columns are already in a group
                    col1_group = [group for group in grouped_columns if col1 in group]
                    col2_group = [group for group in grouped_columns if col2 in group]

                    # If neither column is in a group, create a new group
                    if not col1_group and not col2_group:
                        grouped_columns.append([col1, col2])
                    # If one column is already in a group, add the other column to that group
                    elif col1_group and not col2_group:
                        col1_group[0].append(col2)
                    elif col2_group and not col1_group:
                        col2_group[0].append(col1)
                    elif col1_group != col2_group:
                        # If both columns are in different groups, merge the groups
                        grouped_columns.remove(col1_group[0])
                        grouped_columns.remove(col2_group[0])
                        grouped_columns.append(col1_group[0] + col2_group[0])

                elif 3 < sum(mask) <= 5:
                    # Check if 'دسته بندی' values are the same for both columns
                    category_mask = data.loc[mask, 'دسته بندی'].nunique() == 1

                    if category_mask:
                        # Check if the columns are already in a group
                        col1_group = [group for group in grouped_columns if col1 in group]
                        col2_group = [group for group in grouped_columns if col2 in group]

                        # If neither column is in a group, create a new group
                        if not col1_group and not col2_group:
                            grouped_columns.append([col1, col2])
                        # If one column is already in a group, add the other column to that group
                        elif col1_group and not col2_group:
                            col1_group[0].append(col2)
                        elif col2_group and not col1_group:
                            col2_group[0].append(col1)
                        elif col1_group != col2_group:
                            # If both columns are in different groups, merge the groups
                            grouped_columns.remove(col1_group[0])
                            grouped_columns.remove(col2_group[0])
                            grouped_columns.append(col1_group[0] + col2_group[0])

            return grouped_columns

        grouped_columns = group_similar_columns(similar_columns, self.data)

        def find_common_subsequence(strings):
            # Find common prefix among a list of strings
            common_prefix = os.path.commonprefix(strings)

            # Find common suffix among a list of strings (reversed)
            reversed_strings = [s[::-1] for s in strings]
            common_suffix_reversed = os.path.commonprefix(reversed_strings)[::-1]

            # The common subsequence is the combination of common prefix and common suffix
            common_subsequence = common_prefix + common_suffix_reversed

            if common_subsequence:
                return common_subsequence
            else:
                return None


        def merge_columns_within_groups(data, grouped_columns):
            merged_values = {}

            for group in grouped_columns:
                # Find rows where at least one column in the group has values
                mask = data[group].notnull().any(axis=1)

                # Use the common substring of the column names as the new column name
                common_substring = find_common_subsequence(group)
                new_column_name = f"{common_substring}_Merged" if common_substring else 'Merged'

                # Concatenate values with ' | ' separator, excluding NaN values using apply
                data[new_column_name] = data.loc[mask, group].astype(str).apply(lambda row: ' | '.join(cell for cell in row if cell != 'nan'), axis=1)


                # Drop the original columns
                data.drop(columns=group, inplace=True, errors='ignore')  # Ignore errors if columns don't exist

            # Concatenate all columns at once to improve performance
            data = pd.concat([data, data[new_column_name]], axis=1)
            data.drop(columns=[new_column_name], inplace=True, errors='ignore')

            return data#, merged_values

        # df, merged_values = merge_columns_within_groups(self.data, grouped_columns)
        self.data = merge_columns_within_groups(self.data, grouped_columns)

    def clean_data_in_each_column(self):
        ###Nans cluster manually

        def cluster_column(df, column, k=3):

            # Check if the column contains lists
            df[column] = df[column].apply(lambda x: tuple(x) if isinstance(x, list) else x)

                
            # Convert the column to string to handle non-string entries
            string_data = df[column].astype(str)
            
            # Identify entries that contain at least one numeric character
            mixed_mask = string_data.str.contains('\d')
            
            # Convert mixed entries to numeric values
            numeric_values = pd.to_numeric(df.loc[mixed_mask, column], errors='coerce')
            
            # Use LabelEncoder to convert strings to numerical labels for non-numeric entries
            label_encoder = LabelEncoder()
            encoded_data = label_encoder.fit_transform(string_data[~mixed_mask])
            
            # Reshape the data for K-means
            data = encoded_data.reshape(-1, 1)
            
            # Initialize the KMeans model
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            
            # Fit the model to the non-numeric data
            kmeans.fit(data)
            
            # Add the cluster labels to the DataFrame for non-numeric entries
            df[column + '_cluster_labels'] = -1  # Initialize with -1
            df.loc[~mixed_mask, column + '_cluster_labels'] = kmeans.labels_

        for column in self.data.columns[3:]:
            print(f"====={column}=====")
            # print(f"Column Number: {self.data.columns.get_loc(column)}")
            # self.data=
            cluster_column(self.data, column)
            
            # # Check if cluster -1 has values for the current column
            # cluster_label_column = column + '_cluster_labels'
            # if cluster_label_column in self.data.columns:
            # cluster_minus_one_values = self.data[self.data[cluster_label_column] == -1][column]
            # if not cluster_minus_one_values.empty:
            #     # print("----- There are Unique values in cluster -1: -----")
            #     num_clusters = len(self.data[cluster_label_column].unique()) - 1
                
                # print(f"Column '{column}' values in cluster -1:")
                # if isinstance(cluster_minus_one_values.iloc[0], list):
                #     # Convert lists to tuples for uniqueness
                #     unique_values_cluster_minus_one = cluster_minus_one_values.apply(tuple).unique()
                    # print(unique_values_cluster_minus_one)
                # else:
                #     print(cluster_minus_one_values)
            # else:
            #     num_clusters = len(self.data[cluster_label_column].unique())

        self.data = self.data.drop(self.data.columns[[0, 1]], axis=1)

        

        def set_cluster_data_to_nan_multiple(data, column_indices, cluster_numbers_list):
            updates_dict = dict(zip(column_indices, cluster_numbers_list))
            
            # Create a set to store the data before making it NaN
            data_before_nan = set()

            for update in updates_dict.items():
                column_index, cluster_numbers = update

                for cluster_number in cluster_numbers:
                    # Identify rows with the specified cluster in the specified column using iloc
                    rows_to_update = data[data.iloc[:, column_index + 1228] == cluster_number].index

                    # Store the cell values before making them NaN
                    data_before_nan.update(set(data.loc[rows_to_update, data.columns[column_index]].values))

                    # Set the values in the specified column to NaN for the identified rows
                    data.loc[rows_to_update, data.columns[column_index]] = np.nan

            return data, data_before_nan

        column_indices = [2, 3, 5,6,7,10,13,17,18,30,38,40,47,49,52,53,54,61,62,72,78,79,86,89,92,97,98,106,107,108, 1035]
        cluster_numbers_list = [[1, 2], [2], [1,2],[1,2],[2],[1,2],[1,2],[1,2],[2],[1,2],[1,2],[1],[1,2],[1,2],[1,2],[1,2],[1,2],[1,2],[1,2],[1,2],[1,2],[1,2],[1,2],[1],[1,2],[1,2],[1,2],[1,2],[1,2],[1,2],[0,1,2],[1,2]]

        self.data, data_before_nan = set_cluster_data_to_nan_multiple(self.data, column_indices, cluster_numbers_list)
                
        ### Delete all anomaly data
        def trace_and_set_to_nan(data, data_before_nan):
            # for col in self.data.columns:
            for col in data.columns[2:1229]:
                for index, cell_value in data[col].items():
                    # Check if the cell value is in the set data_before_nan
                    if cell_value in data_before_nan:
                        data.at[index, col] = np.nan

            return data
            
        self.data = trace_and_set_to_nan(self.data, data_before_nan)

        self.data = self.data.drop(self.data.columns[1229:], axis=1)
        ### delete nan columns and nan rows
        self.data = self.data.dropna(axis=1, how='all')  # Drop NaN columns
        self.data = self.data.dropna(axis=0, how='all')  # Drop NaN rows
    
    
    def remove_price_outlier(self):
        self.data = self.data[self.data['price'] >= 10000]
        category_counts = self.data['دسته بندی'].value_counts()
        categories_to_keep = category_counts[category_counts >= 40].index
        # df_filtered = self.data[self.data['دسته بندی'].isin(categories_to_keep)]
        self.data = self.data[self.data['دسته بندی'].isin(categories_to_keep)]
        self.data.dropna(axis=0, how='all', inplace=True)
        self.data.dropna(axis=1, how='all', inplace=True)
        self.data=self.data.reset_index(drop=True)

        def identify_outliers(prices):
            Q1 = prices.quantile(0.25)
            Q3 = prices.quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            # Return a boolean mask for outliers
            outliers_mask = (prices < lower_bound) | (prices > upper_bound)
            
            return outliers_mask

        # Group by 'دسته بندی' and apply the identify_outliers function
        outliers_info = self.data.groupby('دسته بندی')['price'].apply(identify_outliers)
        outlier_indices = np.where(outliers_info)[0]

        self.data = self.data.drop(outlier_indices)
    
    
    
    
    
    def preprocess_data(self):
        # Bring everything together
        self.load_data()

        # Apply preprocessing function to "product_description" column
        self.data["product_description"] = self.data["product_description"].map(self.preprocess_product_description)
        print("===Done1===")


        self.make_dataSet()
        print("===Done2===")


        self.find_and_delete_few_data()
        print("===Done3===")


        self.find_and_delete_similar_column()
        print("===Done4===")


        self.find_and_merge_category_with_similar_columns_and_price()
        print("===Done5===")


        self.delete_columns_less_5data()
        print("===Done6===")


        self.find_similar_category_name_to_merge()
        print("===Done7===")


        self.clean_data_in_each_column()
        print("===Done8===")


        self.remove_price_outlier()
        print("===Done9===")

        
        return self.data
