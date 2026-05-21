import warnings
import numpy as np 
import pandas as pd 
from sklearn import metrics
import category_encoders as ce
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

warnings.filterwarnings('ignore')
class NPKEstimator:
    def __init__(self, data='Nutrient_recommendation.csv'):
        self.df = pd.read_csv(data, header=None)
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        
        self.renameCol()  # Rename columns
        self.preprocess_data()  # Fix Label_N and remove outliers

    def renameCol(self):
        self.df.columns = ['Crop', 'Temperature', 'Humidity', 'Rainfall', 'Label_N', 'Label_P', 'Label_K']
        self.df.drop(self.df.index[:1], inplace=True)

    def preprocess_data(self):
        """Fix zero values in Label_N and remove outliers."""
        # Convert columns to numeric
        self.df[['Temperature', 'Humidity', 'Rainfall', 'Label_N', 'Label_P', 'Label_K']] = \
            self.df[['Temperature', 'Humidity', 'Rainfall', 'Label_N', 'Label_P', 'Label_K']].apply(pd.to_numeric)

        # Fix zero values in Label_N by replacing them with the median
        median_n = self.df[self.df['Label_N'] > 0]['Label_N'].median()
        self.df['Label_N'] = self.df['Label_N'].replace(0, median_n)

        # Remove outliers using IQR
        for col in ['Label_N', 'Label_P', 'Label_K']:
            Q1 = self.df[col].quantile(0.25)
            Q3 = self.df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            self.df = self.df[(self.df[col] >= lower_bound) & (self.df[col] <= upper_bound)]
    
    def cropMapper(self):
        mapping = dict()
        with open("mapped_crops.csv", "w") as fh:
            fh.write("Crops,Key\n")
            for i, crop in enumerate(np.unique(self.df[['Crop']]), 1):
                mapping[crop] = i
                fh.write("%s,%d\n" % (crop, i))
            mapping['NA'] = np.nan
            fh.write("NA,nan")

        ordinal_cols_mapping = [{"col": "Crop", "mapping": mapping}]
        encoder = ce.OrdinalEncoder(cols='Crop', mapping=ordinal_cols_mapping, return_df=True)
        return mapping, encoder
    
    def estimator(self, crop, temp, humidity, rainfall, y_label):
        
        X = self.df.drop(['Label_N', 'Label_P', 'Label_K'], axis=1)
        y = self.df[y_label]

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.20, random_state=42)

        mapping, encoder = self.cropMapper()
        self.X_train = encoder.fit_transform(self.X_train)
        self.X_test = encoder.transform(self.X_test)

        regressor = RandomForestRegressor(n_estimators=50, random_state=0)
        regressor.fit(self.X_train, self.y_train)

        query = [mapping[crop.strip().lower()], temp, humidity, rainfall]
        y_pred = regressor.predict([query])
        return y_pred