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


class ContactNet: #maybe make this class a singleton pattern

    def __init__(self, train: bool):
        self.train = train

    def load_pretrained_model(self):
        return models.load_model('path/to/model/model.h5')
    
    def save_model(self, mdl):
        mdl.save('path/to/mdl')
    
    def train_model(self):
        pass