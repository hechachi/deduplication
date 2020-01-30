import numpy as np
import pandas as pd
import random
import os

from random import shuffle


class LpuIndex:
    """ Class works with input files and extracts and saves
        information about name, address and indexes. """
    def __init__(self, paths, addr_cols, name_cols):
        """
        Parameters
        --------
        paths: list of str
            Paths to load files.
        addr_cols: 2d list of int
            Columns which contain address information.
        name_cols: 2d list of int
            Columns which contain name information.
            
        Attributes
        --------
        paths: str
            Paths to load files.
        indices: list of np.array()
            List of sequence ids for our lpu, which we get from special column in current file.
            If current column does not exist, we use our ids like range(0, shape_of_our_df).
        indices_loc: list of np.array()
            Numbers, which get access to lpu with current indices.
        sep: list
            Format separations in csv files.
        addr_cols: 2d list of int
            Columns for each file which contain info about address.
        name_cols: 2d list of int
            Columns for rach file which contain info about name.            
        test: LpuIndex
            Test lpu_id.
        train: LpuIndex
            Train lpu_id.
        """
        
        self.paths = list(paths)
        self.indices = []                    
        self.indices_loc = []
        self.addr_cols = list(addr_cols)
        self.name_cols = list(name_cols)
        self.test = LpuIndex
        self.train = LpuIndex
        self.sep = []
        self.ext = []
        
    def fill_indices(self, name_id_col='lpu_id'):
        """ Function try to find 'name_id_col' in columns of input file.
            If found, indices get values from this column.
            Else, indices get range from 0 to string number.
            Default name_id_col = 'lpu_id'.
        
        Parameters
        --------
        name_id_col: str
            Name of column id of lpu to find.    
        """
        
        for idx, file in enumerate(self.paths):
            if file.endswith('.csv'):
                ext = 'csv'
            elif file.endswith('.xlsx'):
                ext = 'xlsx'
            else:
                assert False, "Недопустимый формат"
            self.ext.append(ext)

            df = None
            if self.ext[idx] == 'csv':
                read_flag = False
                for sep in [',', ';', '\t']:
                    try:
                        df = pd.read_csv(file, encoding='cp1251', sep=sep)
                        read_flag = True
                        self.sep.append(sep)
                    except:
                        continue
                assert read_flag, "Используйте формат csv, разделители - запятые".format(file)
            elif self.ext[idx] == 'xlsx':
                try:
                    df = pd.read_excel(file)
                    self.sep.append('None')
                except:
                    raise ValueError("Беда")
            
            help_df = df.iloc[:, self.addr_cols[idx] + self.name_cols[idx]]
            help_df.index = range(0, help_df.shape[0])
            
            if name_id_col in df.columns:
                help_df[name_id_col] = df.loc[:, name_id_col]
            df = help_df
            df.dropna(how='any', inplace=True)
            
            self.indices_loc.append(np.array(df.index)) 
            if name_id_col in df.columns:
                self.indices.append(np.array(df[name_id_col].astype(str).values))
            else:
                self.indices.append(np.array(df.index.astype(str)))               
                   
        return self
    
    def cv_split(self, train_size=0.8, shuffle=False):
        """ Function creat 2 object of LpuIndex and split indices on (train and test) 
        
        Parameters
        --------
        train_size: float in [0.0, 1.0]
            Size of train subset.
        shuffle: bool
            If True, indices will be shuffle.
        """
        
        self.train = LpuIndex(self.paths, self.addr_cols, self.name_cols)
        self.test = LpuIndex(self.paths, self.addr_cols, self.name_cols)
        
        if shuffle:
            for i in range(len(self.paths)):
                help_list = list(range(0, self.indices_loc[i].size))
                random.shuffle(help_list)
                help_list = np.asarray(help_list)
                
                help_list_1 = help_list[:int(len(help_list)*train_size)]
                help_list_2 = help_list[int(len(help_list)*train_size)+1:]
                
                self.train.indices_loc.append(
                    self.indices_loc[i][help_list_1]
                )
                self.train.indices.append(
                    self.indices[i][help_list_1]
                ) 
                self.test.indices_loc.append(
                    self.indices_loc[i][help_list_2]
                )
                self.test.indices.append(
                    self.indices[i][help_list_2]
                )
        else:
            for i in range(len(self.paths)):
                self.train.indices_loc.append(
                    self.indices_loc[i][:int(self.indices_loc[i].size*train_size)]
                )
                self.train.indices.append(
                    self.indices[i][:int(self.indices[i].size*train_size)]
                )    
                self.test.indices_loc.append(
                    self.indices_loc[i][int(self.indices_loc[i].size*train_size)+1:]
                )    
                self.test.indices.append(
                    self.indices[i][int(self.indices[i].size*train_size)+1:]
                )
            
        return self    