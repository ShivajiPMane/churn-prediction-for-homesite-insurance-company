import numpy as np
import pandas as pd
import pickle
# import joblib
from datetime import datetime
# from sklearn.preprocessing import OneHotEncoder


class Preprocess:

    def __init__(self):

        self.__default_SF5_cr = 0.05158873442859722
        self.__default_PF9_cr = 0.2278782877483513
        self.__default_SF1B_cr = 0.2970842197472156
        self.__SalesField5_cr_dct = pickle.load(open("files/SalesField5_cr_dict.pkl", 'rb'))
        self.__PersonalField9_cr_dct = pickle.load(open("files/PersonalField9_cr_dict.pkl", 'rb'))
        self.__SalesField1B_cr_dct = pickle.load(open("files/SalesField1B_cr_dict.pkl", 'rb'))
        self.__cat_encoders_dict = pickle.load(open("files/cat_encoders_dict.pkl", 'rb'))
        self.__binary_features = pickle.load(open("files/binary_features.pkl", 'rb'))
        self.__feature_pairs = pickle.load(open("files/feature_pairs.pkl", 'rb'))

    def preprocessing_datapoint(self, data_point):
        # 1 missing value count

        data_point["missing_value_count"] = data_point.isna().sum(axis=1)

        # 2 As per competition Host -1 denotes missing values, So we will count it and add it in "missing_value_count"

        data_point["missing_value_count"] = data_point["missing_value_count"].add(
            data_point[data_point == -1].count().sum())

        # 3 missing value treatment

        data_point.fillna('-1', inplace=True)
        data_point["PersonalField84"] = data_point["PersonalField84"].astype(
            int)  # as "PersonalField84" & "PropertyField29" are originally
        data_point["PropertyField29"] = data_point["PropertyField29"].astype(
            int)  # numerical features, so we have reset it to int

        # 4 removing features which are having only one unique value count, as they won't contribute much in prediction

        data_point.drop(["PropertyField6", "GeographicField10A"], axis=1, inplace=True)

        # 5 check for any discrepancies and remove them

        data_point["Original_Quote_Date"].apply(
            lambda x: datetime.strptime(x, '%Y-%m-%d'))  # checking "Original_Quote_Date"

        if data_point['Field10'].dtype == object:  # "Field10" feature value may have comma ','
            data_point['Field10'] = int(data_point['Field10'].str.replace(',', ''))

        data_point["GeographicField63"] = data_point["GeographicField63"].replace([" "], [
            "N"])  # impute backspace (" ") of "GeographicField63" feature with most frequent value

        # 6 extract day, month, year, dayOfWeek from feature "Original_Quote_Date"

        data_point["Quote_Year"] = pd.to_datetime(data_point["Original_Quote_Date"]).dt.year
        data_point["Quote_Quarter"] = pd.to_datetime(data_point["Original_Quote_Date"]).dt.quarter
        data_point["Quote_Month"] = pd.to_datetime(data_point["Original_Quote_Date"]).dt.month
        data_point["Quote_Day"] = pd.to_datetime(data_point["Original_Quote_Date"]).dt.day
        data_point["Quote_DayOfWeek"] = pd.to_datetime(data_point["Original_Quote_Date"]).dt.dayofweek

        data_point.drop(["Original_Quote_Date"], axis=1, inplace=True)  # dropping "Original_Quote_Date"

        # 7 make new features "#_Negation" & "#_Affirmation"

        df = data_point[self.__binary_features]
        data_point["#_Negation"] = list(df[df.isin(['N', 0])].count(axis=1).values)
        data_point["#_Affirmation"] = list(df[df.isin(['Y', 1])].count(axis=1).values)

        # 8 make new feature "same_value_count"

        data_point["same_value_count"] = self.__same_value_cal(data_point, self.__feature_pairs)

        # 9 Feature Engineering with conversion rate
        data_point['FE_salesField5'] = self.__SalesField5_cr_dct.get(data_point['SalesField5'].values[0],
                                                                     self.__default_SF5_cr)
        data_point['FE_pesonalField9'] = self.__PersonalField9_cr_dct.get(data_point['PersonalField9'].values[0],
                                                                          self.__default_PF9_cr)
        data_point['FE_salesField1B'] = self.__SalesField1B_cr_dct.get(data_point['SalesField1B'].values[0],
                                                                       self.__default_SF1B_cr)

        data_point.drop(['QuoteNumber'], axis=1, inplace=True)  # dropping "QuoteNumber"

        # 10 encoding categorical features using OneHotEncoding

        encoded_data_point = self.__encode_cat_feature(data_point, self.__cat_encoders_dict)

        # dropping original categorical features
        data_point.drop(labels=list(self.__cat_encoders_dict.keys()), axis=1, inplace=True)

        # 11 concating dataframes

        data_point = pd.concat([data_point, encoded_data_point], axis=1)

        return data_point

    def __same_value_cal(self, df, l):

        """
        This function takes a dataframe and list of column names as inputs and returns an array of length
        equal to number of data points in the dataset. Each value of the array stores the number of column
        pairs where the feature values are equal.
        """

        same_val = np.zeros(len(df), dtype=int)

        for i in range(0, len(l), 2):
            df_tmp = df[df[l[i]] == df[l[i + 1]]][df[l[i]] != (-1)][[l[i], l[i + 1]]]
            for j in df_tmp.index:
                same_val[j] += 1

        return same_val

    def __encode_cat_feature(self, dataset, encoder_dict):
        encoded_data = []
        flag = 0
        for column, encoder in encoder_dict.items():
            column_names = encoder.get_feature_names_out([column])
            if flag == 0:
                encoded_data = pd.DataFrame(encoder.transform(dataset[column].values.reshape(-1, 1)).toarray(),
                                            columns=column_names)
                flag = 1
            else:
                encoded_data = encoded_data.join(
                    pd.DataFrame(encoder.transform(dataset[column].values.reshape(-1, 1)).toarray(),
                                 columns=column_names))
        return encoded_data
