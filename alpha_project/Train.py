import os
import re
import pandas as pd
import numpy as np

from fuzzywuzzy import fuzz
from alpha_project.LpuList import LpuList
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score


class Train(object):

    def __init__(self, base, request, k):
        """ Init function.
        
        Parameters
        --------
        A: LpuList
            Base.
        B: LpuList
            Client request.
        k: int
            Top K candidates for each lpu.
            
        Attributes
        --------
        _base: LpuList
            Base.
        _request: LpuList
            Client request.
        _options: list of tuples
            Parameters for choice best lpu.
        _suit_nodes: list
            Nodes from tree.
        _lpu_by_lsh_name: dict
            Best lpu for different options(keys) using name.
        _lpu_by_lsh_addr: dict
            Best lpu for different options(keys) using address.
        _indices_{addr\name}_{union\inter}: list
            Union or intersection _lpu_by_lsh_name or _lpu_by_lsh_addr by on each lpu.
        _N: int
            Parameter for choice algorithm.
        _k: int
            top k.
        candidates: 2D list
            Lpu indices.
        features: dict
            Features for learning.
        tree:
            Model.
        new_candidates: 2D list
             Lpu indices after model working.
        """

        assert isinstance(base, LpuList), "Недопустимый тип первого параметра. Убедитесь, что тип - LpuList"
        assert isinstance(request, LpuList), "Недопустимый тип второго параметра. Убедитесь, что тип - LpuList"

        self._base = base
        self._request = request
        self._options = None
        self._suit_nodes = []
        self._lpu_by_lsh_name = {}
        self._lpu_by_lsh_addr = {}
        self._indices_name_union = []
        self._indices_addr_union = []
        self._indices_name_inter = []
        self._indices_addr_inter = []
        self._N = None
        self._k = k
        self.candidates = [[] for _ in range(len(self._request.name))]
        self.features = {}
        self.tree = None
        self.x_holdout = None
        self.y_holdout = None
        self.x_train = None
        self.y_train = None
        self.new_candidates = None

    def _get_request_process_addresses_list_(self):
        """ Function return list of OrderedDict with address information. """
        return self._request.features['process_addresses_list']

    def _get_suit_node(self, address):
        """ Function return node from our tree implementation. """
        return self._base.tree.suitable_nodes(address)

    def _get_indices_lsh(self, *args, idx=0, n=10):
        """ Function return name and address best indices use lsh. """

        return list(self._base.lsh['{}_name_{}_{}lsh'.format(*args)].query(
                    self._request.features['{}_name_{}_{}minhash'.format(*args)][idx], n)), \
            list(self._base.lsh['{}_addr_{}_{}lsh'.format(*args)].query(
                 self._request.features['{}_addr_{}_{}minhash'.format(*args)][idx], n))

    def _node_level(self, idx):
        """ Function return node level for current lpu. """
        return self._suit_nodes[idx].level

    def _node_indices(self, idx):
        """ Function return indices in current node. """
        return self._suit_nodes[idx].lpu_id

    def _node_parent_indices(self, idx, how):
        """ Function return parent indices for current node.
            (It works with how=1. If how=2, 3 ... we get parent of parent...
            indices for current node). """

        if how == 1:
            return self._suit_nodes[idx].parent.lpu_id
        elif how == 2:
            return self._suit_nodes[idx].parent.parent.lpu_id
        elif how == 3:
            return self._suit_nodes[idx].parent.parent.parent.lpu_id

    def _prepare_indices_name(self):
        """ Function prepare union and interception on name n-grams. """

        request_length = len(self._suit_nodes)
        for idx in range(request_length):
            help_list_1 = list(self._lpu_by_lsh_name[self._options[0]][idx])
            help_list_2 = list(self._lpu_by_lsh_name[self._options[0]][idx])
            for opt in self._options[1:]:
                for i in self._lpu_by_lsh_name[opt][idx]:
                    if i not in help_list_1:
                        help_list_1.append(i)
                for i in self._lpu_by_lsh_name[opt][idx]:
                    if i in help_list_2:
                        help_list_2.remove(i)
            self._indices_name_union.append(help_list_1)
            self._indices_name_inter.append(help_list_2)

        return self

    def _prepare_indices_name_mod(self):
        """ Function prepare union and interception on name n-grams. """

        return self

    def _prepare_indices_addr(self):
        """ Function prepare union and interception on address n-grams. """

        request_length = len(self._suit_nodes)
        for idx in range(request_length):
            help_list_1 = list(self._lpu_by_lsh_addr[self._options[0]][idx])
            help_list_2 = list(self._lpu_by_lsh_addr[self._options[0]][idx])
            for opt in self._options[1:]:
                for i in self._lpu_by_lsh_addr[opt][idx]:
                    if i not in help_list_1:
                        help_list_1.append(i)
                for i in self._lpu_by_lsh_addr[opt][idx]:
                    if i in help_list_2:
                        help_list_2.remove(i)
            self._indices_addr_union.append(help_list_1)
            self._indices_addr_inter.append(help_list_2)

        return self

    def _prepare_best_lpu(self):
        """ Function fill _suit_nodes, _lpu_by_lsh_name, _lpu_by_lsh_addr, _suit_options. """

        request_process_addresses_list = self._get_request_process_addresses_list_()

        # fill _suit_nodes
        for my_dict in request_process_addresses_list:
            suit_node = self._get_suit_node(my_dict)
            self._suit_nodes.append(suit_node)

        # fill _indices
        for opt in self._options:

            self._lpu_by_lsh_name[opt] = []
            self._lpu_by_lsh_addr[opt] = []

            for idx, my_dict in enumerate(request_process_addresses_list):
                lpu_by_lsh_name, lpu_by_lsh_addr = self._get_indices_lsh(*opt, idx=idx, n=self._N)

                self._lpu_by_lsh_name[opt].append(lpu_by_lsh_name)
                self._lpu_by_lsh_addr[opt].append(lpu_by_lsh_addr)

        self._prepare_indices_name()
        self._prepare_indices_addr()

        return self

    def prepare_candidates(self, n=10, options=None):
        """ Function prepare _candidates

        Parameters
        --------
        n: int
            It's a number that use in top N for lsh_forest.
        options: list
            list of ( , , ) - parameters for get best lpu.
        """

        self._N = n
        self._options = options or [
            ('weighed', 'word', 1),
            ('not_weighed', 'word', 1),
            ('not_weighed', 'char', 3)
        ]

        self._prepare_best_lpu()

        request_length = len(self._suit_nodes)

        for idx in range(request_length):

            # Совпадение блока или дома
            if self._node_level(idx) >= 3:

                self.candidates[idx] = list(
                    [k for k in self._suit_nodes[idx].lpu_id if k in self._indices_name_inter[idx]]
                )
                for k in self._suit_nodes[idx].lpu_id:
                    if k not in self.candidates[idx]:
                        self.candidates[idx].append(k)

                how = 1 if self._node_level(idx) == 3 else 2

                tree = self._node_parent_indices(idx=idx, how=how+1)
                for i in self._indices_name_union[idx]:   # мб union
                    if (i in tree) and (i not in self.candidates[idx]):
                        self.candidates[idx].append(i)

                tree = self._node_parent_indices(idx=idx, how=how)

                for k in tree:
                    if k not in self.candidates[idx]:
                        self.candidates[idx].append(k)

            # Совпадение с точностью до улицы.
            elif self._node_level(idx) == 2:
                self.candidates[idx] = list(
                    [k for k in self._suit_nodes[idx].lpu_id if k in self._indices_name_union[idx]]
                )

                tree = self._node_parent_indices(idx=idx, how=1)
                for i in self._indices_name_union[idx]:   # мб union
                    if (i in tree) and (i not in self.candidates[idx]):
                        self.candidates[idx].append(i)

                for k in self._suit_nodes[idx].lpu_id:
                    if k not in self.candidates[idx]:
                        self.candidates[idx].append(k)

            # Остальное
            else:
                self.candidates[idx] = list(
                    [elem for elem in self._indices_addr_inter[idx] if elem in self._indices_name_union[idx]]
                )

            for i in self._indices_name_inter[idx]:
                if i not in self.candidates[idx]:
                    self.candidates[idx].append(i)

            for k in self._indices_name_union[idx]:
                if k not in self.candidates[idx]:
                    self.candidates[idx].append(k)

            self.candidates[idx] = list(self.candidates[idx][:self._k])

        return self

    @staticmethod
    def names_by_indices(lpulist, indices):
        """ Function return names by indices from base. """
        return [lpulist.name[np.where(lpulist.indices == idx)[0][0]] for idx in indices if idx != -1]

    @staticmethod
    def address_by_indices(lpulist, indices):
        """ Function return address by indices from base. """
        return [lpulist.addr[np.where(lpulist.indices == idx)[0][0]] for idx in indices if idx != -1]

    @staticmethod
    def smth_by_indices(lpulist, indices, smth='city'):
        """ Function return something(from dict) by indices from base. """
        return [lpulist.features[smth][np.where(lpulist.indices == idx)[0][0]] for idx in indices if idx != -1]

    def address_feature_prepare(self):
        """ Function prepare binary features related with address.
            There are equal: city, street, house, house_mod, block, address. """

        self.features['city'] = []
        self.features['street'] = []
        self.features['house'] = []
        self.features['house_mod'] = []
        self.features['block'] = []
        self.features['address'] = []

        for idx, i in enumerate(self._request.name):

            for smth in ['city', 'street', 'house', 'block']:
                for j in self.smth_by_indices(self._base, self.candidates[idx], smth=smth):
                    self.features[smth].append(self._request.features[smth][idx] == j)

            # В номере дома оставляем только цифры.
            for idx_1, j in enumerate(self.smth_by_indices(self._base, self.candidates[idx], smth='house')):
                try:
                    d_1 = re.sub(r'\D', '', self._request.features['house'][idx])
                except:
                    d_1 = -1
                try:
                    d_2 = re.sub(r'\D', '', j)
                except:
                    d_2 = -2
                self.features['house_mod'].append(d_1 == d_2)

                self.features['address'].append(
                    self.features['city'][idx_1] and self.features['street'][idx_1] and
                    (self.features['house'][idx_1] or self.features['house_mod'][idx_1]) and self.features['block'][
                        idx_1]  # Можно ослабить, не добавляя последнее.
                )

        return self

    def word_to_vec_cos_prepare(self):
        """ Function prepare cos similarity on word_to_vec vectors. """

        self.features['w2v_cos_name'] = []
        self.features['w2v_cos_addr'] = []

        for idx, i in enumerate(self._request.features['w2v_name']):

            for j in self.smth_by_indices(self._base, self.candidates[idx], smth="w2v_name"):
                val = (np.linalg.norm(i) * np.linalg.norm(j))
                if val:
                    self.features['w2v_cos_name'].append(np.dot(i, j) / (np.linalg.norm(i) * np.linalg.norm(j)))
                if not val:
                    self.features['w2v_cos_name'].append(0)

            for j in self.smth_by_indices(self._base, self.candidates[idx], smth="w2v_addr"):
                val = (np.linalg.norm(i) * np.linalg.norm(j))
                if val:
                    self.features['w2v_cos_addr'].append(np.dot(i, j) / (np.linalg.norm(i) * np.linalg.norm(j)))
                if not val:
                    self.features['w2v_cos_addr'].append(0)

        return self

    def answer_prepare(self, file_path):
        """ Function prepare answer. """

        file_path = file_path or "C:/Users/Илья/Desktop/work/alpha_medicine/Uniq_разметка.xlsx"
        df = pd.read_excel(file_path)
        id_ = df['id'].values

        self.features['answer'] = []

        for idx, i in enumerate(self._request.name):
            for j in self.candidates[idx]:
                if j != -1:
                    self.features['answer'].append(id_[idx] == int(j))

        return self

    def levenstein_distance_prepare(self):
        """ Function prepare levenstein distance on names and addresses. """

        self.features['levenstein_by_name'] = []
        self.features['levenstein_by_addr'] = []

        for idx, i in enumerate(self._request.name):
            for j in self.names_by_indices(self._base, self.candidates[idx]):
                self.features['levenstein_by_name'].append(fuzz.ratio(
                    self._request.name[idx], j)
                )
            for j in self.address_by_indices(self._base, self.candidates[idx]):
                self.features['levenstein_by_addr'].append(fuzz.ratio(
                    self._request.addr[idx], j)
                )

        return self

    def number_in_name(self):
        """ Function prepare additional features (number in name, equal_number_in_name). """

        self.features['number_in_name'] = []
        self.features['equal_number_in_name'] = []

        for idx, i in enumerate(self._request.name):
            for j in self.names_by_indices(self._base, self.candidates[idx]):
                number_1 = re.sub(r'\D', '', i)
                number_2 = re.sub(r'\D', '', j)
                val_1 = 1 if number_1 and number_2 else 0
                val_2 = 1 if number_1 == number_2 else 0
                self.features['number_in_name'].append(val_1)
                self.features['equal_number_in_name'].append(val_2)

        return self

    def jaccard_distance_preprare(self):
        """ Function prepare jaccard distance on names and addresses. """

        options = [
            'not_weighed_name_word_1',
            'not_weighed_name_char_3',
            'not_weighed_addr_word_1',
            'not_weighed_addr_char_3',
        ]
        for opt in options:
            self.features[opt] = []

        for opt in options:
            for idx, i in enumerate(self._request.name):
                for j in self.smth_by_indices(self._base, self.candidates[idx], smth='{}minhash'.format(opt)):
                    self.features[opt].append(self._request.features['{}minhash'.format(opt)][idx].jaccard(j))

        return self

    def _results_on_metrics(self, answer_file_path=None, top=10):
        """ Return count of finding lpu in top, where top is parameter"""

        uniq_df = pd.read_excel(answer_file_path)
        uniq_id = uniq_df['id'].values

        self.sum_top = 0
        for idx, i in enumerate(self.new_candidates):
            if str(uniq_id[idx]) in self.new_candidates[idx][:top]:
                self.sum_top += 1

        return self.sum_top

    def create_model(self, answer_file_path="data/request/file_for_model",
                     test_size=0.5, max_depth=5):
        """ Function create model. """

        self.address_feature_prepare()
        self.word_to_vec_cos_prepare()
        self.answer_prepare(file_path=answer_file_path)
        self.levenstein_distance_prepare()
        self.number_in_name()
        self.jaccard_distance_preprare()

        self.tree = RandomForestClassifier(max_depth=max_depth, random_state=17)

        x = pd.DataFrame(self.features)
        y = x.loc[:, 'answer']
        x = x.drop(['answer'], axis=1)

        self.x_train, self.x_holdout, self.y_train, self.y_holdout = train_test_split(
            x, y, test_size=test_size, random_state=17, shuffle=False
        )

        self.tree.fit(self.x_train, self.y_train)

        predict_proba = self.tree.predict_proba(x)[:, 0]

        l_bord = 0
        r_bord = 0
        self.new_candidates = []
        for idx, i in enumerate(self.candidates):
            r_bord += len(self.candidates[idx])
            val = list(zip(i, predict_proba[l_bord:r_bord]))
            xs = sorted(val, key=lambda tup: tup[1])
            self.new_candidates.append([x[0] for x in xs])
            l_bord = r_bord

        self._results_on_metrics(answer_file_path)

        # scores = cross_val_score(tree, x_train, y_train, cv = 10, scoring='precision')
        # scores

        return self
