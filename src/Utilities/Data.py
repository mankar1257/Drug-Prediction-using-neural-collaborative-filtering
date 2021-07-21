# For loading and preprossing the data

import pandas as pd
import scipy.sparse as sp
import numpy as np
import random


class Data:

    def __init__(self):

        self.df = pd.read_csv(
            '/home/vaibhav/Downloads/archive/drugsComTrain_raw.csv').drop(['uniqueID', 'date'], axis=1)
        self.Medical_Conditions_name_to_ID = dict()
        self.Drugs_name_to_ID = dict()
        self.Medical_Conditions_ID_to_NAME = dict()
        self.Drugs_ID_to_NAME = dict()

    def Set_mapping(self):

        Medical_Conditions = self.df['condition'].unique()
        Drugs = self.df['drugName'].unique()

        Medical_Conditions_ID_to_name = dict()

        for i in range(len(Medical_Conditions)):
            key = i
            value = Medical_Conditions[i]
            Medical_Conditions_ID_to_name[key] = value

        Drugs_ID_to_name = {}

        for i in range(len(Drugs)):
            key = i
            value = Drugs[i]
            Drugs_ID_to_name[key] = value

        self.Medical_Conditions_name_to_ID = dict(
            [(value, key) for key, value in Medical_Conditions_ID_to_name.items()])
        self.Drugs_name_to_ID = dict(
            [(value, key) for key, value in Drugs_ID_to_name.items()])

        self.Medical_Conditions_ID_to_NAME = pd.DataFrame(
            list(Medical_Conditions_ID_to_name.items()))
        self.Drugs_ID_to_NAME = pd.DataFrame(list(Drugs_ID_to_name.items()))

    def Get_mapping(self, Medical_Conditions=False, Drugs=False):

        self.Set_mapping()
        if Medical_Conditions == True:
            return [self.Medical_Conditions_ID_to_NAME,
                    self.Medical_Conditions_name_to_ID]

        if Drugs == True:
            return[self.Drugs_ID_to_NAME, self.Drugs_name_to_ID]

        return [self.Medical_Conditions_ID_to_NAME,
                self.Medical_Conditions_name_to_ID, self.Drugs_ID_to_NAME,
                self.Drugs_name_to_ID]

    def Get_data(self):

        self.Set_mapping()

        self.df = self.df[['condition', 'drugName', 'usefulCount']].copy()
        for i in range(len(self.df['drugName'])):
            self.df['drugName'][i] = self.Drugs_name_to_ID[self.df['drugName'][i]]

        for i in range(len(self.df['condition'])):
            self.df['condition'][i] = self.Medical_Conditions_name_to_ID[self.df['condition'][i]]

        self.df.sort_values("condition", axis=0, ascending=True,
                            inplace=True)

        train = sp.dok_matrix((885, 3436), dtype=np.float32)

        for i in range(len(self.df['condition'])):
            ls = list(self.df.iloc[i])
            train[ls[0], ls[1]] = 1.0

        test = []

        for j in range(200):
            i = random.randint(0, 161296)
            ls = list(self.df.iloc[i])
            test.append([ls[0], ls[1]])

        return train, test
