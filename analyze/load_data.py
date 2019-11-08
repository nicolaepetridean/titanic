from numpy import genfromtxt
import os
import pandas as pd
import numpy as np



class TitanicDataLoad:

    def __init__(self):
        self.test_data={}
        self.test_header_columns=[]
        self.train_data={}
        self.submission = {}
        self.train_header_columns = []
        self.train_header_columns_X = []
        self.test_header_columns_X = []
        self.train_header_columns_Y = []
        self.train_data_X = {}
        self.test_data_X = {}
        self.train_data_Y = {}

    def loadFile(self, path):
        self.test_data = pd.read_csv(os.path.join(path, 'test.csv'), delimiter=',')
        self.train_data = pd.read_csv(os.path.join(path, 'train.csv'), delimiter=',')
        self.submission = pd.read_csv(os.path.join(path, 'gender_submission.csv'), delimiter=',')

        self.test_header_columns = [i for i in self.test_data.columns]
        self.train_header_columns = [i for i in self.train_data.columns]
        self.test_data = self.test_data.to_numpy()
        self.train_data = self.train_data.to_numpy()

        self.train_header_columns_Y = [index for index, column in enumerate(self.train_header_columns) if column == "Survived"]
        self.train_header_columns_X = [index for index, column in enumerate(self.train_header_columns)
                                       if column != "Survived" and column != "Name" and column != "PassengerId"
                                       and column != "Ticket" and column != 'Cabin' and column != 'Embarked']

        self.test_header_columns_X = [index for index, column in enumerate(self.test_header_columns)
                                       if column != "Name" and column != "PassengerId"
                                       and column != "Ticket" and column != 'Cabin' and column != 'Embarked']

        #exclude name column
        self.train_data_X = self.train_data[:, [i for i in self.train_header_columns_X]]
        self.train_data_Y = self.train_data[:, [i for i in self.train_header_columns_Y]]

        self.test_data_X = self.test_data[:, [i for i in self.test_header_columns_X]]

        print('self.test_data' + str(self.test_data.shape))
        print('self.train_data' + str(self.train_data.shape))