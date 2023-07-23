import os
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import MinMaxScaler, label_binarize
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import InstanceHardnessThreshold
from keras import models
from keras.models import Sequential
from keras.layers import Input, Dense, Dropout
from keras.callbacks import EarlyStopping
from keras.utils import to_categorical




class ContactNet:

    def __init__(self, train_mode: bool):
        self.training_mode = train_mode # if False -> inference

    def load_pretrained_model(self):
        return models.load_model('model/model.keras')
    
    def save_model(self, mdl) -> None:
        mdl.save('model/model.keras')



    def load_data_ring(self, path: str) -> pd.DataFrame:
        dfs = []
        for filename in os.listdir(path + 'data/features_ring'):
            if filename[-4:] == '.tsv':
                dfs.append(pd.read_csv(path + 'data/features_ring/' + filename, sep='\t'))
        df = pd.concat(dfs)

        return df

    def preprocess_data(self, df: pd.DataFrame):
        df = df[df.Interaction.notna()]
        contact_dict = {"HBOND": 0, "IONIC": 1, "PICATION": 2, "PIPISTACK": 3, "SSBOND": 4, "VDW": 5}
        y = df['Interaction'].copy()

        cat_names = list(y.astype('category').cat.categories)
        y.replace(contact_dict, inplace=True)

        X = df[['s_up', 's_down', 's_phi', 's_psi', 's_a1', 's_a2', 's_a3', 's_a4', 's_a5', 't_up', 't_down', 't_phi', 't_psi', 't_a1', 't_a2', 't_a3', 't_a4', 't_a5']]
        X = X.apply(lambda x: x.fillna(x.mean()) if x.dtype.kind in 'biufc' else x)
        
        #############################################################################
        # All features are selected anyways so Best Subset Selection is not necessary

        #feature_sel = SelectFromModel(LogisticRegression(max_iter=100000))
        #X= feature_sel.fit(X, y)

        #############################################################################

        minMax = MinMaxScaler()
        minMax.fit(X)
        X_scaled = minMax.transform(X)

        return X_scaled, y, cat_names
    

    def balance_data(self, X_train, y_train):
        logging.info("Applying Oversampling...")
        oversample = SMOTE(sampling_strategy={2:20000,3:10000,4:10000})
        X_bal, y_bal = oversample.fit_resample(X_train, y_train)

        return X_bal, y_bal
    
    

    def init_model(self, input_dim, num_classes) -> Sequential:
        model = Sequential()

        model.add(Input(input_dim))

        model.add(Dense(units=128, activation='relu', kernel_initializer="glorot_normal"))
        model.add(Dense(units=128, activation='relu', kernel_initializer="glorot_normal"))
        model.add(Dense(units=128, activation='relu', kernel_initializer="glorot_normal"))
        model.add(Dense(units=128, activation='relu', kernel_initializer="glorot_normal"))
        model.add(Dense(units=128, activation='relu', kernel_initializer="glorot_normal"))

        model.add(Dense(units=num_classes, activation='softmax', kernel_initializer="glorot_normal"))
        
        model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['AUC'])
        
        return model


    def train(self, model, X_bal, y_bal, y_cat, kfold, early_stopping):
        fold = 0
        hist = []
        for train_idx, val_idx in kfold.split(X_bal, y_bal):
            fold += 1
            print(f"Fold {fold}/10")
            Xfold_train, Xfold_val = X_bal[train_idx], X_bal[val_idx]
            yfold_cat_train, yfold_cat_val = y_cat[train_idx], y_cat[val_idx]
            metrics = model.fit(Xfold_train, yfold_cat_train,
                    validation_data=(Xfold_val, yfold_cat_val),
                    epochs=500, verbose=1,
                    batch_size=16000,
                    callbacks=[early_stopping])
            hist.append(metrics)
        
        return hist #model should already be passed by reference so I just return hist


    def report(self, model, X_test, y_test, num_classes):
        outputs = model.predict(X_test)
        y_pred = np.argmax(outputs, axis=1)
        y_true = y_test

        accuracy = accuracy_score(y_true, y_pred)
        logging.info(f"Accuracy: {accuracy}")

        # Binarize labels and compute ROC AUC
        y_true_bin = label_binarize(y_true, classes=np.arange(num_classes))
        y_pred_bin = label_binarize(y_pred, classes=np.arange(num_classes))
        auc = roc_auc_score(y_true_bin, y_pred_bin, average='macro', multi_class='ovr')
        logging.info(f"AUC: {auc}")

        report = classification_report(y_true, y_pred)
        cm = confusion_matrix(y_true, y_pred)

        # Plotting confusion matrix
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, cmap="Blues", fmt="d")
        plt.title("Confusion Matrix")
        plt.xlabel("Predicted Labels")
        plt.ylabel("True Labels")
        plt.show()

        return report


    def train_model(self, path=''):
        df = self.load_data_ring(path)
        X_scaled, y, cat_names = self.preprocess_data(df)

        input_dim = X_scaled.shape[1]
        num_classes = len(cat_names)

        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, stratify=y, test_size=0.1, random_state=100)
        
        kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=100)



        X_train = np.array(X_train)
        y_train = np.array(y_train)
        X_test = np.array(X_test)
        y_test = np.array(y_test)

        X_bal, y_bal = self.balance_data(X_train, y_train)

        y_cat = to_categorical(y_bal, num_classes)

        model = self.init_model(input_dim, num_classes)

        logging.info("Model summary:")
        model.summary()

        es = EarlyStopping(
                       monitor='loss',
                       mode='min',
                       patience=5,
                       min_delta=0.0001
                       )
        
        hist = self.train(model, X_bal, y_bal, y_cat, kf, es)

        self.report(model, X_test, y_test, num_classes)

        return model, hist


