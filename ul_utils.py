import os, shutil, traceback
import kagglehub
import pandas as pd
import itertools
from sklearn.model_selection import cross_validate
from tqdm.auto import tqdm
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import json

class Report_Data():
    def __init__(
            self,
            root='./', 
            demo_size=0.1, 
            cancer_source='zahidmughal2343/global-cancer-patients-2015-2024',
            bankruptcy_source='fedesoriano/company-bankruptcy-prediction'
            ):
        '''
        Inputs:
        - root: home directory (default ./)
        - demo_size: portion of full data to save as demo (default 0.1)
        - cancer_source: kaggle source for cancer data
        - bankruptcy_source: kaggle source for bankruptcy data
        '''
        self.root = root
        self.demo_size = demo_size
        self.cancer_source = cancer_source
        self.bankruptcy_source = bankruptcy_source
        self.latest_loaded = None  # latest data source loaded

    @staticmethod
    def copy_csvs(source_dir, dest_path):
        '''
        Copy only csvs with shutil
        '''
        for path, subpaths, filenames in os.walk(source_dir):
            for filename in filenames:
                if filename.endswith('.csv'):
                    source_path = os.path.join(path, filename)
                    shutil.copy(source_path, dest_path)
        return dest_path

    def download_kaggle_dataset(self, data_dir, save_path):
        '''
        Use the kaggle API to download a dataset
        '''
        tmp_path = kagglehub.dataset_download(data_dir)
        moved_to = self.copy_csvs(tmp_path, save_path)
        return moved_to

    def set_up_data(self):
        '''
        One-stop-shop for setting up our data from kaggle.
        
        Must have kaggle.json saved to the required directory, per kaggle's documentation: 
        - https://www.kaggle.com/docs/api
        '''

        # set up data folders
        if not os.path.exists(self.root+'data/'):
            os.mkdir(self.root+'data/')
            os.mkdir(self.root+'data/cancer/')
            os.mkdir(self.root+'data/cancer/full/')
            os.mkdir(self.root+'data/cancer/demo/')
            os.mkdir(self.root+'data/bankruptcy/')
            os.mkdir(self.root+'data/bankruptcy/full/')
            os.mkdir(self.root+'data/bankruptcy/demo/')

        # set up log folders
        if not os.path.exists(self.root+'logs/'):
            os.mkdir(self.root+'logs/')

        # download the cancer data
        if not os.path.exists(self.root+'data/cancer/full/data.csv'):
            self.download_kaggle_dataset(
                'zahidmughal2343/global-cancer-patients-2015-2024',
                self.root+'data/cancer/full/data.csv'
                )
            print('Cancer Dataset Downloaded')
        else:
            print('Already have Cancer Dataset')
        
        # # set up cancer demo
        if not os.path.exists(self.root+'data/cancer/demo/data.csv'):
            sample_df = pd.read_csv(self.root+'data/cancer/full/data.csv')
            sample_idx = sample_df.sample(round(sample_df.shape[0]*self.demo_size)).index
            sample_df.loc[sample_idx].reset_index().to_csv(self.root+'data/cancer/demo/data.csv')
        print('Cancer Demo Saved')

        # download the bankruptcy data
        if not os.path.exists(self.root+'data/bankruptcy/full/data.csv'):
            self.download_kaggle_dataset(
                'fedesoriano/company-bankruptcy-prediction',
                self.root+'data/bankruptcy/full/data.csv'
                )
            print('Bankruptcy Dataset Downloaded')
        else:
            print('Already Have Bankruptcy Dataset')

        # set up bankruptcy demo
        # samle each label evenly 
        if not os.path.exists(self.root+'data/bankruptcy/demo/data.csv'):
            sample_df = pd.read_csv(self.root+'data/bankruptcy/full/data.csv')
            sample_1s = sample_df[sample_df['Bankrupt?']==1].reset_index()
            sample_1s_idx = sample_1s.sample(round(sample_1s.shape[0]*self.demo_size)).index
            sample_1s = sample_1s.loc[sample_1s_idx]
            sample_0s = sample_df[sample_df['Bankrupt?']==0].reset_index()
            sample_0s_idx = sample_0s.sample(round(sample_0s.shape[0]*self.demo_size)).index
            sample_0s = sample_0s.loc[sample_0s_idx]
            sample_df = pd.concat([sample_1s, sample_0s]).reset_index()
            sample_df.to_csv(self.root+'data/bankruptcy/demo/data.csv')
        print('Bankruptcy Demo Saved')

    @staticmethod
    def remove_bad_cols(data):
        try:
            data = data.drop(['Unnamed: 0', 'index'], axis=1)
            return data
        except:
            return data

    def get_cancer_full(self):
        self.latest_loaded = 'full_cancer'
        return self.remove_bad_cols(pd.read_csv(self.root+'data/cancer/full/data.csv'))

    def get_cancer_demo(self):
        self.latest_loaded = 'demo_cancer'
        return self.remove_bad_cols(pd.read_csv(self.root+'data/cancer/demo/data.csv'))

    def get_bankruptcy_full(self):
        self.latest_loaded = 'full_bankruptcy'
        return self.remove_bad_cols(pd.read_csv(self.root+'data/bankruptcy/full/data.csv'))

    def get_bankruptcy_demo(self):
        self.latest_loaded = 'demo_bankruptcy'
        return self.remove_bad_cols(pd.read_csv(self.root+'data/bankruptcy/demo/data.csv'))

if __name__=='__main__':
    datasource = Report_Data()
    datasource.set_up_data()