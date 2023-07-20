from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import InstanceHardnessThreshold
from keras.models import Sequential
from keras.layers import Input, Dense, Dropout
from keras.callbacks import EarlyStopping
from keras.utils import to_categorical
from keras import models
import pandas as pd
import numpy as np
import os
import logging


class ContactNet: #maybe make this class a singleton pattern

    def __init__(self, train: bool):
        self.train = train

    def load_pretrained_model(self):
        return models.load_model('path/to/model/model.h5')
    
    def save_model(self, mdl) -> None:
        mdl.save('path/to/mdl')



    def load_data_ring(path: str) -> pd.DataFrame:
        dfs = []
        for filename in os.listdir(path + 'data/features_ring'):
            if filename[-4:] == '.tsv':
                dfs.append(pd.read_csv(path + 'data/features_ring/' + filename, sep='\t'))
        df = pd.concat(dfs)

        return df

    def preprocess_data(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        df = df[df.Interaction.notna()]
        contact_dict = {"HBOND": 0, "IONIC": 1, "PICATION": 2, "PIPISTACK": 3, "SSBOND": 4, "VDW": 5}
        y = df['Interaction']
        cat_names = list(y.astype('category').cat.categories)
        y.replace(contact_dict, inplace=True)
        X = df[['s_up', 's_down', 's_phi', 's_psi', 's_a1', 's_a2', 's_a3', 's_a4', 's_a5', 't_up', 't_down', 't_phi', 't_psi', 't_a1', 't_a2', 't_a3', 't_a4', 't_a5']]
        X = X.apply(lambda x: x.fillna(x.mean()) if x.dtype.kind in 'biufc' else x)
        
        return X, y, cat_names


    def train_model(self, path):
        df = self.load_data_ring(path)
        X, y, cat_names = self.preprocess_data(df)

        num_classes = len(cat_names)

        # Is it selecting all features? Useless?
        #feature_sel = SelectFromModel(LogisticRegression(max_iter=100000))
        #feature_sel.fit(X, y)

        minMax = MinMaxScaler()
        minMax.fit(X)
        X_scaled = minMax.transform(X)

        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, stratify=y, test_size=0.1, random_state=100)
        
        kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=100)

        X_train = np.array(X_train)
        y_train = np.array(y_train)
        X_test = np.array(X_test)
        y_test = np.array(y_test)


        logging.info("Applying Undersampling...")
        undersample = InstanceHardnessThreshold(estimator=AdaBoostClassifier(),sampling_strategy={0:70000,5:80000})
        X_bal, y_bal = undersample.fit_resample(X_train, y_train)
        logging.info("Applying Oversampling...")
        oversample = SMOTE(sampling_strategy={1:20000,3:10000,2:20000,4:10000})
        X_bal, y_bal = oversample.fit_resample(X_bal, y_bal)

        y_cat = to_categorical(y_bal, num_classes)