
# coding: utf-8

# In[ ]:

import numpy as np
import pandas as pd
import string
import nltk
import requests
import re
import gensim

from dadata import DaDataClient
from sklearn.feature_extraction.text import TfidfVectorizer
from datasketch import MinHash, LeanMinHash, MinHashLSHForest, WeightedMinHashGenerator
from alpha_project.Tree import Node, Tree
from collections import OrderedDict


class LpuList:
    
    lsh = {}
    vocab = {} 
    
    def __init__(self, index):        
        """
        
        Parameters
        --------
        index: LpuIndex
            Contain information from files (file names, indices, position indices, ...).
            For more, see class LpuIndex.
        
        Attributes
        --------
        tree: Tree
            For present our process addresses in own class Tree.        
        index: LpuIndex
            Contain information from files (file names, indices, position indices, ...).
            For more, see class LpuIndex.
        addr: np.array() of str
            Clinics addresses.
        name: np.array() of str
            Clinics names.
        indices: np.array() of str
            Clinics indices.          
        features: dict
            Features for data.
        fmt: optional, 'csv' or 'npz'
            Format of load data.
        """
        
        self.tree = Tree()
        self.index = index
        
        self.indices = np.empty(0)
        self.addr = np.empty(0)
        self.name = np.empty(0)
        
        self.features = {}
        
        self.fmt = None
        self.n_char = []
        self.n_word = []
        self.num_perm = None
        self.options = None
        
    def __getitem__(self, key):

        a = []
        for feature in list(self.features.keys()):
            a.append(self.features[feature][key])  
        return np.array(a)
        
    def load(self, fmt, npz_file_path=None):
        """ Load function.
        
        Parameters
        --------
        fmt: optional: 'csv' or 'npz'
            Format of input file. Function work with raw (csv) and processed (npz) data.
            When use 'csv', function creat data from our csv file. When use 'npz',
            function load data and features from 'npz' file. Default, fmt = 'csv'.
        npz_file_path: str
            File with data in 'npz' format. Usable, when fmt = 'npz'.
        """
        
        self.fmt = fmt
        
        if fmt == 'csv':  
            for idx, file in enumerate(self.index.paths):                
                if self.index.ext[idx] == 'csv':
                    df = pd.read_csv(file, encoding='cp1251', sep=self.index.sep[idx])
                elif self.index.ext[idx] == 'xlsx':
                    df = pd.read_excel(file)
            
                help_df = df.iloc[:, self.index.addr_cols[idx]].astype(str)            
                self.addr = np.append(self.addr, np.array([', '.join(help_df.values[i])
                                                           for i in self.index.indices_loc[idx]]))
                
                help_df = df.iloc[:, self.index.name_cols[idx]].astype(str)
                self.name = np.append(self.name, np.array([', '.join(help_df.values[i])
                                                           for i in self.index.indices_loc[idx]]))
                
                self.indices = np.append(self.indices, self.index.indices[idx])
            
        elif fmt == 'npz':
            
            # Проверка неизменности исходной базы (?). 
            
            file_flag = npz_file_path is None
            assert not file_flag, "Не указан путь к файлу."
            
            try:
                loaded = np.load(npz_file_path)
            except:
                raise ValueError("Некорректный формат файла.")
                
            self.addr = loaded['addr']
            self.name = loaded['name']
            self.indices = loaded['indices']
            self.features['process_addresses_list'] = loaded['process_addresses_list']
            self.features['location'] = loaded['location']
            
        return self
    
    def dump(self, file_path):
        """ Save data into a single file in compressed .npz format.
        
        Parameters
        --------
        file_path: str
            File name to save.
        
        """
        
        np.savez_compressed(
            file_path, indices=self.indices, 
            addr=self.addr, name=self.name,
            process_addresses_list=self.features['process_addresses_list'], 
            location=[]            
        )
        
        return self
    
    def clear_name_and_address(self):
        """ Function clear name and address """

        self.name = [re.sub(r'\W', ' ', i) for i in self.name]
        self.name = [i.strip() for i in self.name]
        self.name = [re.sub(r'\s+', ' ', i) for i in self.name]
        self.name = [i.strip() for i in self.name]
        self.name = [i.lower() for i in self.name]

        self.addr = [re.sub(r'\W', ' ', i) for i in self.addr]
        self.addr = [i.strip() for i in self.addr]
        self.addr = [re.sub(r'\s+', ' ', i) for i in self.addr]
        self.addr = [i.strip() for i in self.addr]
        self.addr = [i.lower() for i in self.addr]

        return self
    
    def dadata_request(self, key='', secret=''):
        """ Function doing requests to dadata.ru and return addresses information.
            Servise has limit - 10k/day. """
        
        # Если используются сохраненные данные, то ничего не происходит.
        if self.fmt == 'npz':
            return self
        
        # Иначе данные заполняются из запроса.
        client = DaDataClient(
            key=key,
            secret=secret
        )
        
        self.features['location'] = []
        self.features['process_addresses_list'] = []
        self.features['postal_code'] = []
        
        for i in range(self.addr.size):
            client.suggest_address = self.addr[i]
            client.suggest_address.request()
            
            try:
                help_dict = OrderedDict()
                for key in ['city', 'street', 'house', 'block']:
                    help_dict.update({key: client.result.suggestions[0]['data'][key]})  
                    
                if help_dict['city'] is None:
                    help_dict.update({'city': client.result.suggestions[0]['data']['settlement']})                                     
                self.features['process_addresses_list'].append(help_dict)
                
            except:
                self.features['process_addresses_list'].append({})
                
            try:
                self.features['postal_code'].append(client.result.suggestions[0]['data']['postal_code'])
            except:
                self.features['postal_code'].append([])
        
        return self
    
    def address_features_extaction(self):
        """ """
        
        self.features['city'] = []
        self.features['street'] = []
        self.features['house'] = []
        self.features['block'] = []
        
        for i in self.features['process_addresses_list']:
            try:
                self.features['city'].append(i['city'])
                self.features['street'].append(i['street'])
                self.features['house'].append(i['house'])
                self.features['block'].append(i['block'])
            except:
                self.features['city'].append(None)
                self.features['street'].append(None)
                self.features['house'].append(None)
                self.features['block'].append(None)
            
        return self    
    
    def to_tree(self):
        """ Function present processed addresses in tree structure. """
        
        for idx, my_dict in enumerate(self.features['process_addresses_list']):
            self.tree.add_elem(my_dict, self.indices[idx])
            
        return self
    
    @staticmethod
    def names_by_indices(lpulist, indices):
        # Возможно улучшить поиск?
        return [lpulist.name[np.where(lpulist.indices == idx)[0][0]] for idx in indices]
    
    def w2v(self, path_to_model, use_components=None):
        """
        Each word will has numerical vector in some space. Function use model, trained on russian wiki.
        
        Parameters
        --------
        path_to_model: str
            Path to model file.
        use_components: str, optional: ['name', 'addr'] or ['name'] or ['addr'].
            Components to transform.
        """
        
        use_components = use_components or ['name']
        
        try:
            model = gensim.models.KeyedVectors.load_word2vec_format(path_to_model, binary=True)
        except:
            assert False, "Не удалось загрузить модель."
        
        # Проверка типа.
        use_components = list(use_components)
               
        if ('name' in use_components and 'addr' in use_components and len(use_components) > 2) or       \
           ('name' in use_components and 'addr' not in use_components and len(use_components) > 1) or  \
           ('name' not in use_components and 'addr' in use_components and len(use_components) > 1) or  \
           ('name' not in use_components and 'addr' not in use_components):
                assert False, "Проверьте значение параметра use_components."
        
        # Чуток переделываем словарь. Убираем части речи.
        my_dict = {}
        for key in model.vocab.keys():
            new_key, _ = key.split('_')
            my_dict[new_key] = model.vocab[key]
        model.vocab = my_dict
        
        for i in use_components:
            self.features['w2v_{}'.format(i)] = []
            for idx, tokens in enumerate(self.features['token_list_{}'.format(i)]):
                count = 0
                result = np.array(0)
                
                for token in tokens:
                    try:        
                        result = result + np.array(model[token])
                        count += 1
                    except:
                        pass
            
                if count >= 2:
                    result /= count
                   
                self.features['w2v_{}'.format(i)].append(result)
        
        return self
        
    def change_symb(self, template=None, change_on=None, use_components=None):
        """ Function change template on other template for name and\or address.
        
        Parameters
        --------        
        use_components: str, optional: ['name', 'addr'] or ['name'] or ['addr'].
            Components to use.
        template: list of str.
            Сharacters sequence that change.
        change_on: list of str.
            Characters sequence for changing.
        """
        
        if not template:
            return self
        if not change_on:
            change_on = ['']*len(template)
        
        use_components = use_components or ['name']
        if 'name' not in use_components and 'addr' not in use_components:
            assert False, "Проверьте значение параметра use_components."
        
        for component in use_components:
            value = self.name if component == 'name' else self.addr   
            for idx, i in enumerate(value):
                value[idx] = value[idx].lower()
                for pos, _ in enumerate(template):
                    value[idx] = value[idx].replace(template[pos], change_on[pos])
                
        return self        
    
    def ngrams(self, use_components=None, type_option=None, n_char=None, n_word=None):
        """ Function create diffirent form of ngrams.
        
        Parameters
        --------
        use_components: str, optional: ['name', 'addr'] or ['name'] or ['addr'].
            Components to use.
        type_option: str, optional: ['char', 'word'] or ['char'] or ['word'].
            Components to use.
        n_char: list of int
            sizes of char grams. 
        n_word: list of int
            sizes of word grams.
        """
        
        use_components = use_components or ['name']
        type_option = type_option or ['char']
        n_char = n_char or [3]
        n_word = n_word or [1]
        
        if 'char' not in type_option and 'word' not in type_option:
            assert False, "Проверьте значение параметра type_option."
            
        if 'name' not in use_components and 'addr' not in use_components:
            assert False, "Проверьте значение параметра use_components."    
        
        self.n_char = list(n_char)
        self.n_word = list(n_word)
        
        for i in use_components:
            if 'char' in type_option:                 
                for n in n_char:
                    self.features['{}_char_{}grams'.format(i, n)] = [[name[i:i+n].lower().replace(' ', '_') for i
                                                                     in range(len(name)-n+1) if len(name) >= n]
                                                                     for name in self.name]

            if 'word' in type_option:
                for n in n_word:
                    self.features['{}_word_{}grams'.format(i, n)] = \
                        [[''.join(tokens[i:i+n]) for i in range(len(tokens)-n+1) if len(tokens) >= n]
                         for tokens in self.features['token_list_{}'.format(i)]]
        return self

    def get_tf_idf_vectors(self, use_components=None, type_option=None,
                           n_char=None, n_word=None, isrequest=False, npz=False):
        """ Function create diffirent form of ngrams.
        
        Parameters
        --------
        use_components: str, optional: ['name', 'addr'] or ['name'] or ['addr'].
            Components to use.
        type_option: str, optional: ['char', 'word'] or ['char'] or ['word'].
            Components to use.
        n_char: list of int
            sizes of char grams. 
        n_word: list of int
            sizes of word grams.
        isrequest: bool
            If this LpuList created for base, we have False.
        npz: bool
            Indicator or using npz files.
        """
        
        use_components = use_components or ['name']
        type_option = type_option or ['char']
        n_char = n_char or [3]
        n_word = n_word or [1]
        
        if 'char' not in type_option and 'word' not in type_option:
            assert False, "Проверьте значение параметра type_option."
            
        if 'name' not in use_components and 'addr' not in use_components:
            assert False, "Проверьте значение параметра use_components."    
        
        self.n_char = list(n_char)
        self.n_word = list(n_word)
        token_pattern = '(?u)\\b\\w+\\b'
        
        for component in use_components:
            value = self.name if component == 'name' else self.addr                
            for option in type_option:
                n_list = n_char if option == 'char' else n_word                   
                for n in n_list: 
                    if isrequest:
                        vocabulary = np.load(
                            'data/vocabulary_dadata/{}_{}_{}.npy'.format(component, option, n)
                        ).item()
                        
                        td = TfidfVectorizer(
                            analyzer=option, ngram_range=(n, n),
                            vocabulary=vocabulary,
                            token_pattern=token_pattern
                        )
                    else:
                        td = TfidfVectorizer(analyzer=option, ngram_range=(n, n),
                                             token_pattern=token_pattern)
                        
                    x = td.fit_transform(value)
                    if not isrequest:
                        np.save('data/vocabulary_dadata/{}_{}_{}.npy'.format(component, option, n),
                                td.vocabulary_)
                    if not npz:    
                        self.features['tf_idf_{}_{}_{}grams'.format(component, option, n)] = x.toarray()
            
        return self

    def lsh_forest(self, algoritm_type=None, use_components=None, type_option=None, n_char=None, n_word=None):
        """
        LSH Function.
        
        Parameters
        --------
        use_components: list, optional: ['name', 'addr'] or ['name'] or ['addr'].
            Components to use.
        type_option: list, optional: ['char', 'word'] or ['char'] or ['word'].
            Components to use.
        n_char: list of int
            sizes of char grams. 
        n_word: list of int
            sizes of word grams.
        algoritm_type: list, optional: [weighed] or [not_weighed]
            Type of algorithm
        """
        
        algoritm_type = algoritm_type or 'not_weighed'
        use_components = use_components or ['name']
        type_option = type_option or ['char']
        n_char = n_char or [3]
        n_word = n_word or [1]
        
        if 'char' not in type_option and 'word' not in type_option:
            assert False, "Проверьте значение параметра type_option."
            
        if 'name' not in use_components and 'addr' not in use_components:
            assert False, "Проверьте значение параметра use_components." 
        
        for i in use_components:
            for j in type_option:
                n_list = n_char if j == 'char' else n_word                    
                for n in n_list:
                    LpuList.lsh['{}_{}_{}_{}lsh'.format(algoritm_type, i, j, n)] = MinHashLSHForest(self.num_perm)
        
                    for idx, minhash in enumerate(self.features['{}_{}_{}_{}minhash'.format(algoritm_type, i, j, n)]):
                        LpuList.lsh['{}_{}_{}_{}lsh'.format(algoritm_type, i, j, n)].add(self.indices[idx], minhash)
        
                    LpuList.lsh['{}_{}_{}_{}lsh'.format(algoritm_type, i, j, n)].index()   
        
        return self
    
    def min_hash(self, num_perm=64, seed=42, use_components=None,
                 type_option=None, n_char=None, n_word=None, npz=None, isrequest=False):
        """
        Minhash function.
        
        Parameters
        --------
        num_perm: int
            Number of permutations.
        seed: int
            For random permutations.
        use_components: str, optional: ['name', 'addr'] or ['name'] or ['addr'].
            Components to use.
        type_option: str, optional: ['char', 'word'] or ['char'] or ['word'].
            Components to use.
        n_char: list of int
            sizes of char grams. 
        n_word: list of int
            sizes of word grams.
        isrequest: bool
            If this LpuList created for base, we have False.
        npz: bool
            Indicator or using npz files.
        
        """
        
        if npz:           
            self.options = type_option
            self.num_perm = num_perm
            
            n = n_char if type_option == 'char' else n_word
            self.features['not_weighed_{}_{}_{}minhash'.format(use_components[0], type_option[0], n[0])] = \
                np.load(npz)['min_hash']
            return self
        
        use_components = use_components or ['name']
        type_option = type_option or ['char']
        n_char = n_char or [3]
        n_word = n_word or [1]
        
        if 'char' not in type_option and 'word' not in type_option:
            assert False, "Проверьте значение параметра type_option."
            
        if 'name' not in use_components and 'addr' not in use_components:
            assert False, "Проверьте значение параметра use_components."         
        
        self.options = type_option
        self.num_perm = num_perm
        
        for i in use_components:
            for j in type_option:
                n_list = n_char if j == 'char' else n_word                   
                for n in n_list:
                    help_list = []
                    for idx, name in enumerate(self.features['{}_{}_{}grams'.format(i, j, n)]):
                        minhash = MinHash(num_perm, seed=seed)
                        for ngram in name:
                            minhash.update(ngram.encode('utf8'))
                        lean_minhash = LeanMinHash(minhash)    
                        help_list.append(lean_minhash)
        
                    self.features['not_weighed_{}_{}_{}minhash'.format(i, j, n)] = np.array(help_list)
                    file_path = 'data/min_hash_dadata/{}_{}_{}_not_weighed_minhash.npz'.format(i, j, n)
                    if not isrequest:
                        np.savez_compressed(
                            file_path, min_hash=np.array(help_list)
                        )
                    
        return self
    
    def weighed_min_hash(self, num_perm=64, seed=42, use_components=None, 
                         type_option=None, n_char=None, n_word=None, npz=None, isrequest=False):
        """ """
        
        if npz:
            self.options = type_option
            self.num_perm = num_perm
            
            n = n_char if type_option == 'char' else n_word
            self.features['weighed_{}_{}_{}minhash'.format(use_components[0], type_option[0], n[0])] = \
                np.load(npz)['min_hash']
            return self
        
        use_components = use_components or ['name']
        type_option = type_option or ['char']
        n_char = n_char or [3]
        n_word = n_word or [1]
        
        if 'char' not in type_option and 'word' not in type_option:
            assert False, "Проверьте значение параметра type_option."
            
        if 'name' not in use_components and 'addr' not in use_components:
            assert False, "Проверьте значение параметра use_components."
            
        self.options = type_option
        self.num_perm = num_perm
        
        for i in use_components:
            for j in type_option:               
                n_list = n_char if j == 'char' else n_word                   
                for n in n_list:
                    wmg = WeightedMinHashGenerator(
                        len(self.features['tf_idf_{}_{}_{}grams'.format(i, j, n)][0]),
                        sample_size=num_perm, seed=seed
                    )
                    help_list = []
                    for vector in self.features['tf_idf_{}_{}_{}grams'.format(i, j, n)]:
                        if np.all(vector == 0):
                            vector[0] = 0.000001  # Это костылек.
                        help_list.append(wmg.minhash(vector))
                    self.features['weighed_{}_{}_{}minhash'.format(i, j, n)] = np.array(help_list)
                    file_path = 'data/min_hash_dadata/{}_{}_{}_weighed_minhash.npz'.format(i, j, n)
                    if not isrequest:
                        np.savez_compressed(
                            file_path, min_hash=np.array(help_list)
                        )
                    
                    del self.features['tf_idf_{}_{}_{}grams'.format(i, j, n)]
                    
        return self
        
    def ya_request(self):
        """ """

        # Если используются сохраненные данные, то ничего не происходит.
        if self.fmt == 'npz':
            return self
        
        url = 'https://geocode-maps.yandex.ru/1.x/?format=json&results=1&geocode='

        self.features['location'] = []
        self.features['process_addresses_list'] = []
        self.features['postal_code'] = []
        
        for i in self.addr:
            response = requests.get(url+i.replace(' ', '+'))           
            
            try:
                info_list = response.json()['response']['GeoObjectCollection'] \
                                           ['featureMember'][0]['GeoObject'] \
                                           ['metaDataProperty']['GeocoderMetaData'] \
                                           ['Address']['Components']
                my_dict = OrderedDict()                                                                   
            except:
                info_list = []
                
            try:
                postal_code = response.json()['response']['GeoObjectCollection'] \
                                                         ['featureMember'][0]['GeoObject'] \
                                                         ['metaDataProperty']['GeocoderMetaData'] \
                                                         ['Address']['postal_code']
            except:
                postal_code = []
                
            for k in info_list:
                if k['kind'] == 'locality':
                    my_dict.update({'city': k['name']})
                if k['kind'] == 'street':
                    my_dict.update({'street': k['name']})
                if k['kind'] == 'house':
                    my_dict.update({'house': k['name']})
            
            try:            
                my_location = OrderedDict()
                
                location = response.json()['response']['GeoObjectCollection']['featureMember'] \
                                          [0]['GeoObject']['Point']['pos'].split(' ')
                my_location.update({'geo_lat': location[1]})
                my_location.update({'geo_lon': location[0]})
            except:
                my_location = OrderedDict({'geo_lat': 0, 'geo_lon': 0})
                
            self.features['location'].append(my_location)
            self.features['process_addresses_list'].append(my_dict)
            self.features['postal_code'].append(postal_code)
                                          
        return self                            
    
    def get_token_list(self, use_components=None):
        """ Function split eath string on tokens.
        
        Parameters
        --------
        use_components: str, optional: ['name', 'addr'] or ['name'] or ['addr'].
            Components to use.
        """
        
        use_components = use_components or ['name']        
        if 'name' not in use_components and 'addr' not in use_components:
            assert False, "Проверьте значение параметра use_components."         
        
        for component in use_components:
            value = self.name if component == 'name' else self.addr
            self.features['token_list_{}'.format(component)] = [
                [''.join([s for s in k if s not in string.punctuation]).lower()
                 for k in nltk.word_tokenize(i)]
                for i in value
            ]
            self.features['token_list_{}'.format(component)] = [
                [k for k in name if k != ''] for name in self.features['token_list_{}'.format(component)]
            ]
            
        return self
    
    def get_length(self):
        """ Function get length for all objects. """
        
        self.features['name_length'] = np.array([len(i) for i in self.name])
        self.features['addr_length'] = np.array([len(i) for i in self.addr])
        return self

    def prepare_min_hash_and_lsh(self, B, algoritm_types=None, type_options=None, n_char=None, n_word=None,
                                 use_components=None):
        """ :) """

        algoritm_types = algoritm_types or None
        type_options = type_options or None
        n_char = n_char or None
        n_word = n_word or None
        use_components = use_components or None

        for algoritm_type in algoritm_types:
            for type_option in type_options:
                n_list = n_char if type_option == 'char' else n_word
                for n in n_list:
                    for use_component in use_components:
                        if algoritm_type == 'weighed':
                            (self.get_tf_idf_vectors(
                                use_components=[use_component],
                                type_option=[type_option],
                                n_char=[n], n_word=[n], isrequest=False, npz=True
                            ).weighed_min_hash(
                                num_perm=256, seed=42, use_components=[use_component],
                                type_option=[type_option],
                                n_char=[n], n_word=[n],
                                npz='data/min_hash_dadata/{}_{}_{}_weighed_minhash.npz'.format(use_component, type_option, n)
                            ).lsh_forest(
                                algoritm_type=algoritm_type,
                                use_components=[use_component],
                                type_option=[type_option],
                                n_char=[n], n_word=[n]
                            ))

                            (B.get_tf_idf_vectors(
                                use_components=[use_component],
                                type_option=[type_option],
                                n_char=[n], n_word=[n], isrequest=True, npz=False
                            ).weighed_min_hash(
                                num_perm=256, seed=42,
                                use_components=[use_component],
                                type_option=[type_option],
                                n_char=[n], n_word=[n], isrequest=True
                            ))
                        else:
                            (self.ngrams(
                                use_components=[use_component],
                                type_option=[type_option],
                                n_char=[n], n_word=[n]
                            ).min_hash(
                                num_perm=256, seed=42,
                                use_components=[use_component],
                                type_option=[type_option],
                                n_char=[n], n_word=[n],
                                npz='data/min_hash_dadata/{}_{}_{}_not_weighed_minhash.npz'.format(use_component,
                                                                                              type_option, n)
                            ).lsh_forest(
                                algoritm_type=algoritm_type,
                                use_components=[use_component],
                                type_option=[type_option],
                                n_char=[n], n_word=[n]
                            ))

                            (B.ngrams(
                                use_components=[use_component],
                                type_option=[type_option],
                                n_char=[n], n_word=[n],
                            ).min_hash(
                                num_perm=256, seed=42,
                                use_components=[use_component],
                                type_option=[type_option],
                                n_char=[n], n_word=[n], isrequest=True
                            ))