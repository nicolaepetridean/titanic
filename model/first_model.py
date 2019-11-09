from sklearn.preprocessing import StandardScaler

from keras.models import Sequential
from keras.layers import Dense
from sklearn.preprocessing import LabelEncoder
import numpy as np
import pandas as pd


def labelEncoder(X_train, column_to_encode):
    label_encoder_sex = LabelEncoder()
    X_train[:,column_to_encode] = label_encoder_sex.fit_transform(X_train[:,column_to_encode])

    return X_train


def normalize_data(X_train):
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)

    return X_train


def replace_Nan_With_Zero(X_train):
    return np.nan_to_num(X_train)


def create_model_and_train(X_train, Y_train, X_test):

    X_train = normalize_data(X_train)
    X_train = replace_Nan_With_Zero(X_train)

    X_test = normalize_data(X_test)
    X_test = replace_Nan_With_Zero(X_test)

    classifier = Sequential()
    # Input layer with 5 inputs neurons
    classifier.add(Dense(output_dim=5, init='uniform', activation='relu', input_dim=5))
    # Hidden layer
    classifier.add(Dense(output_dim=5, init='uniform', activation='relu'))
    # output layer with 1 output neuron which will predict 1 or 0
    classifier.add(Dense(output_dim=1, init='uniform', activation='sigmoid'))

    classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    classifier.fit(X_train, Y_train, batch_size=5, nb_epoch=150)

    # getting predictions of test data
    prediction = classifier.predict(X_test).tolist()
    # list to series
    se = pd.Series(prediction)

    return se
