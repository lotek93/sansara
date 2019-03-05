import os
import time
import numpy as np
import matplotlib as mpl
import pickle
import copy
import gc

import tensorflow as tf
import keras
from keras.utils import multi_gpu_model

#from keras.utils.vis_utils import model_to_dot
#SVG(model_to_dot(model, show_shapes=True).create(prog='dot', format='svg'))

# TODO: better mutations (select layer and then brute force parameters)
# TODO: compound blocks: inception, residuals etc
# TODO: intercept GPU OOM
# TODO: dedicated functions mutate_insert, mutate_remove, mutate_change
# TODO: visualizing nnets in dnas
# TODO: check base_dna

layers_list = ['Conv2D', 'Dropout', 'SpatialDropout2D', 'SeparableConv2D', 'Conv2DTranspose', 'UpSampling2D', 'MaxPooling2D',
               'AveragePooling2D', 'GlobalMaxPooling2D', 'LocallyConnected2D', 'Add', 'Concatenate', 'ReLU', 'LeakyReLU',
               'PReLU', 'ELU', 'Softmax', 'Tanh', 'BatchNormalization', 'Dense', 'Flatten', 'DepthwiseConv2D', 'GlobalMaxPooling2D']

layer_params = {
                'Conv2D':             {'filters': lambda: np.random.choice([2**x for x in range(9)]),
                                       'kernel_size': lambda: np.random.randint(1, 10),
                                       'strides': lambda: np.random.randint(1, 5),
                                       'padding': lambda: np.random.choice(["'same'", "'valid'"]),
                                       'use_bias': lambda: np.random.choice([True, False])},
                'DepthwiseConv2D':    {'kernel_size': lambda: np.random.randint(1, 10),
                                       'strides': lambda: np.random.randint(1, 5),
                                       'padding': lambda: np.random.choice(["'same'", "'valid'"]),
                                       'use_bias': lambda: np.random.choice([True, False])},
                'Dropout':            {'rate': lambda: np.random.random()},
                'SpatialDropout2D':   {'rate': lambda: np.random.random()},
                'SeparableConv2D':    {'filters': lambda: np.random.choice([2**x for x in range(9)]),
                                       'kernel_size': lambda: np.random.randint(1, 10),
                                       'padding': lambda: np.random.choice(["'same'", "'valid'"])},
                'Conv2DTranspose':    {'filters': lambda: np.random.choice([2**x for x in range(9)]),
                                       'kernel_size': lambda: np.random.randint(1, 10),
                                       'padding': lambda: np.random.choice(["'same'", "'valid'"])},
                'UpSampling2D':       {'size': lambda: np.random.randint(2, 5)},
                'MaxPooling2D':       {'pool_size': lambda: np.random.randint(2, 5)},
                'AveragePooling2D':   {'pool_size': lambda: np.random.randint(2, 5)},
                'GlobalMaxPooling2D': {},
                'GlobalAveragePooling2D': {},
                'LocallyConnected2D': {'filters': lambda: np.random.choice([2**x for x in range(9)]),
                                       'kernel_size': lambda: np.random.randint(1, 10),
                                       'padding': lambda: np.random.choice(["'same'", "'valid'"])},
                'Add':                {'connect2': lambda: np.random.randint(-25, -1)},
                'Concatenate':        {'connect2': lambda: np.random.randint(-25, -1)},
                'ReLU':               {},
                'LeakyReLU':          {'alpha': lambda: np.random.random()},
                'PReLU':              {},
                'ELU':                {'alpha': lambda: np.random.random()},
                'Softmax':            {},
                'BatchNormalization': {},
                'Dense':              {'units': lambda: np.random.choice([2**x for x in range(9)]),
                                       'activation': lambda: np.random.choice(["'relu'", "'sigmoid'", "'softmax'", "'tanh'"])},
                'Flatten':            {}
                }
transition = {}
transition['Input'] = {'Conv2D': True, 'Dropout': True, 'SpatialDropout2D': True, 'SeparableConv2D': True, 'Conv2DTranspose': True,
                       'UpSampling2D': True, 'MaxPooling2D': True, 'AveragePooling2D': True, 'GlobalMaxPooling2D': True, 'LocallyConnected2D': True,
                       'Add': False, 'Concatenate': False, 'ReLU': True, 'LeakyReLU': True, 'PReLU': True, 'ELU': True, 'Softmax': True,
                       'BatchNormalization': True, 'Dense': True, 'Flatten': True, 'DepthwiseConv2D': True,
                       'GlobalAveragePooling2D': True}
transition['Conv2D'] = {'Conv2D': True, 'Dropout': True, 'SpatialDropout2D': True, 'SeparableConv2D': True, 'Conv2DTranspose': True,
                        'UpSampling2D': True, 'MaxPooling2D': False, 'AveragePooling2D': False, 'GlobalMaxPooling2D': True, 'LocallyConnected2D': True,
                        'Add': True, 'Concatenate': True, 'ReLU': True, 'LeakyReLU': True, 'PReLU': True, 'ELU': True, 'Softmax': True,
                        'BatchNormalization': True, 'Dense': True, 'Flatten': True, 'DepthwiseConv2D': True,
                        'GlobalAveragePooling2D': True}
transition['DepthwiseConv2D'] = {'Conv2D': True, 'Dropout': True, 'SpatialDropout2D': True, 'SeparableConv2D': True, 'Conv2DTranspose': True,
                        'UpSampling2D': True, 'MaxPooling2D': False, 'AveragePooling2D': False, 'GlobalMaxPooling2D': True, 'LocallyConnected2D': True,
                        'Add': True, 'Concatenate': True, 'ReLU': True, 'LeakyReLU': True, 'PReLU': True, 'ELU': True, 'Softmax': True,
                        'BatchNormalization': True, 'Dense': True, 'Flatten': True, 'DepthwiseConv2D': True,
                        'GlobalAveragePooling2D': True}
transition['Dropout'] = {'Conv2D': True, 'Dropout': True, 'SpatialDropout2D': True, 'SeparableConv2D': True, 'Conv2DTranspose': True,
                         'UpSampling2D': True, 'MaxPooling2D': True, 'AveragePooling2D': True, 'GlobalMaxPooling2D': True, 'LocallyConnected2D': True,
                         'Add': True, 'Concatenate': True, 'ReLU': True, 'LeakyReLU': True, 'PReLU': True, 'ELU': True, 'Softmax': True,
                         'BatchNormalization': True, 'Dense': True, 'Flatten': True, 'DepthwiseConv2D': True,
                         'GlobalAveragePooling2D': True}
transition['SpatialDropout2D'] = {'Conv2D': True, 'Dropout': True, 'SpatialDropout2D': True, 'SeparableConv2D': True,
                                  'Conv2DTranspose': True, 'UpSampling2D': True, 'MaxPooling2D': True, 'AveragePooling2D': False, 'GlobalMaxPooling2D': True,
                                  'LocallyConnected2D': True, 'Add': True, 'Concatenate': True, 'ReLU': True, 'LeakyReLU': True, 'PReLU': True,
                                  'ELU': True, 'Softmax': True, 'BatchNormalization': True, 'Dense': True, 'Flatten': True,
                                  'DepthwiseConv2D': True, 'GlobalAveragePooling2D': True}
transition['SeparableConv2D'] = {'Conv2D': True, 'Dropout': True, 'SpatialDropout2D': True, 'SeparableConv2D': True,
                                 'Conv2DTranspose': True, 'UpSampling2D': True, 'MaxPooling2D': True, 'AveragePooling2D': True, 'GlobalMaxPooling2D': True,
                                 'LocallyConnected2D': True, 'Add': True, 'Concatenate': True, 'ReLU': True, 'LeakyReLU': True, 'PReLU': True,
                                 'ELU': True, 'Softmax': True, 'BatchNormalization': True, 'Dense': True, 'Flatten': True,
                                 'DepthwiseConv2D': True, 'GlobalAveragePooling2D': True}
transition['Conv2DTranspose'] = {'Conv2D': True, 'Dropout': True, 'SpatialDropout2D': True, 'SeparableConv2D': True,
                                 'Conv2DTranspose': True, 'UpSampling2D': True, 'MaxPooling2D': True, 'AveragePooling2D': True, 'GlobalMaxPooling2D': True,
                                 'LocallyConnected2D': True, 'Add': True, 'Concatenate': True, 'ReLU': True, 'LeakyReLU': True, 'PReLU': True,
                                 'ELU': True, 'Softmax': True, 'BatchNormalization': True, 'Dense': True, 'Flatten': True,
                                 'DepthwiseConv2D': True, 'GlobalAveragePooling2D': True}
transition['UpSampling2D'] = {'Conv2D': True, 'Dropout': True, 'SpatialDropout2D': True, 'SeparableConv2D': True,
                              'Conv2DTranspose': True, 'UpSampling2D': True, 'MaxPooling2D': True, 'AveragePooling2D': True, 'GlobalMaxPooling2D': True,
                              'LocallyConnected2D': True, 'Add': True, 'Concatenate': True, 'ReLU': True, 'LeakyReLU': True, 'PReLU': True,
                              'ELU': True, 'Softmax': True, 'BatchNormalization': True, 'Dense': True, 'Flatten': True,
                              'DepthwiseConv2D': True, 'GlobalAveragePooling2D': True}
transition['MaxPooling2D'] = {'Conv2D': True, 'Dropout': True, 'SpatialDropout2D': True, 'SeparableConv2D': True,
                              'Conv2DTranspose': True, 'UpSampling2D': True, 'MaxPooling2D': True, 'AveragePooling2D': True, 'GlobalMaxPooling2D': True,
                              'LocallyConnected2D': False, 'Add': True, 'Concatenate': True, 'ReLU': True, 'LeakyReLU': True, 'PReLU': True,
                              'ELU': True, 'Softmax': True, 'BatchNormalization': True, 'Dense': True, 'Flatten': True,
                              'DepthwiseConv2D': True, 'GlobalAveragePooling2D': True}
transition['AveragePooling2D'] = {'Conv2D': True, 'Dropout': True, 'SpatialDropout2D': True, 'SeparableConv2D': True,
                                  'Conv2DTranspose': True, 'UpSampling2D': True, 'MaxPooling2D': True, 'AveragePooling2D': False, 'GlobalMaxPooling2D': True,
                                  'LocallyConnected2D': False, 'Add': True, 'Concatenate': True, 'ReLU': True, 'LeakyReLU': True, 'PReLU': True,
                                  'ELU': True, 'Softmax': True, 'BatchNormalization': True, 'Dense': True, 'Flatten': True,
                                  'DepthwiseConv2D': True, 'GlobalAveragePooling2D': True}
transition['GlobalMaxPooling2D'] = {'Conv2D': False, 'Dropout': True, 'SpatialDropout2D': False, 'SeparableConv2D': False,
                                    'Conv2DTranspose': False, 'UpSampling2D': False, 'MaxPooling2D': False, 'AveragePooling2D': False,
                                    'GlobalMaxPooling2D': False, 'LocallyConnected2D': False, 'Add': True, 'Concatenate': True, 'ReLU': True,
                                    'LeakyReLU': True, 'PReLU': True, 'ELU': True, 'Softmax': True, 'BatchNormalization': True,
                                    'Dense': True, 'Flatten': True, 'DepthwiseConv2D': False, 'GlobalAveragePooling2D': True}
transition['GlobalAveragePooling2D'] = {'Conv2D': False, 'Dropout': True, 'SpatialDropout2D': False, 'SeparableConv2D': False,
                                    'Conv2DTranspose': False, 'UpSampling2D': False, 'MaxPooling2D': False, 'AveragePooling2D': False,
                                    'GlobalMaxPooling2D': True, 'LocallyConnected2D': False, 'Add': True, 'Concatenate': True, 'ReLU': True,
                                    'LeakyReLU': True, 'PReLU': True, 'ELU': True, 'Softmax': True, 'BatchNormalization': True,
                                    'Dense': True, 'Flatten': True, 'DepthwiseConv2D': False, 'GlobalAveragePooling2D': False}
transition['LocallyConnected2D'] = {'Conv2D': True, 'Dropout': True, 'SpatialDropout2D': True, 'SeparableConv2D': True,
                                    'Conv2DTranspose': True, 'UpSampling2D': True, 'MaxPooling2D': True, 'AveragePooling2D': True, 'GlobalMaxPooling2D': True,
                                    'LocallyConnected2D': True, 'Add': False, 'Concatenate': False, 'ReLU': True, 'LeakyReLU': True, 'PReLU': True,
                                    'ELU': True, 'Softmax': True, 'BatchNormalization': True, 'Dense': True, 'Flatten': True,
                                    'DepthwiseConv2D': True, 'GlobalAveragePooling2D': True}
transition['Add'] = {'Conv2D': True, 'Dropout': True, 'SpatialDropout2D': True, 'SeparableConv2D': True, 'Conv2DTranspose': True,
                     'UpSampling2D': True, 'MaxPooling2D': True, 'AveragePooling2D': True, 'GlobalMaxPooling2D': True, 'LocallyConnected2D': True,
                     'Add': True, 'Concatenate': True, 'ReLU': True, 'LeakyReLU': True, 'PReLU': True, 'ELU': True, 'Softmax': True,
                     'BatchNormalization': True, 'Dense': True, 'Flatten': True, 'DepthwiseConv2D': True,
                     'GlobalAveragePooling2D': True}
transition['Concatenate'] = {'Conv2D': True, 'Dropout': True, 'SpatialDropout2D': True, 'SeparableConv2D': True, 'Conv2DTranspose': True,
                            'UpSampling2D': True, 'MaxPooling2D': True, 'AveragePooling2D': True, 'GlobalMaxPooling2D': True, 'LocallyConnected2D': True,
                            'Add': True, 'Concatenate': True, 'ReLU': True, 'LeakyReLU': True, 'PReLU': True, 'ELU': True, 'Softmax': True,
                            'BatchNormalization': True, 'Dense': True, 'Flatten': True, 'DepthwiseConv2D': True,
                            'GlobalAveragePooling2D': True}
transition['ReLU'] = {'Conv2D': True, 'Dropout': True, 'SpatialDropout2D': True, 'SeparableConv2D': True, 'Conv2DTranspose': True,
                      'UpSampling2D': True, 'MaxPooling2D': True, 'AveragePooling2D': True, 'GlobalMaxPooling2D': True, 'LocallyConnected2D': True,
                      'Add': True, 'Concatenate': True, 'ReLU': True, 'LeakyReLU': True, 'PReLU': True, 'ELU': True, 'Softmax': True,
                      'BatchNormalization': True, 'Dense': True, 'Flatten': True, 'DepthwiseConv2D': True,
                      'GlobalAveragePooling2D': True}
transition['LeakyReLU'] = {'Conv2D': True, 'Dropout': True, 'SpatialDropout2D': True, 'SeparableConv2D': True,
                           'Conv2DTranspose': True, 'UpSampling2D': True, 'MaxPooling2D': True, 'AveragePooling2D': True, 'GlobalMaxPooling2D': True,
                           'LocallyConnected2D': True, 'Add': True, 'Concatenate': True, 'ReLU': True, 'LeakyReLU': True, 'PReLU': True,
                           'ELU': True, 'Softmax': True, 'BatchNormalization': True, 'Dense': True, 'Flatten': True,
                           'DepthwiseConv2D': True, 'GlobalAveragePooling2D': True}
transition['PReLU'] = {'Conv2D': True, 'Dropout': True, 'SpatialDropout2D': True, 'SeparableConv2D': True,
                       'Conv2DTranspose': True, 'UpSampling2D': True, 'MaxPooling2D': True, 'AveragePooling2D': True, 'GlobalMaxPooling2D': True,
                       'LocallyConnected2D': True, 'Add': True, 'Concatenate': True, 'ReLU': True, 'LeakyReLU': True, 'PReLU': True, 'ELU': True,
                       'Softmax': True, 'BatchNormalization': True, 'Dense': True, 'Flatten': True,
                       'DepthwiseConv2D': True, 'GlobalAveragePooling2D': True}
transition['ELU'] = {'Conv2D': True, 'Dropout': True, 'SpatialDropout2D': True, 'SeparableConv2D': True,
                     'Conv2DTranspose': True, 'UpSampling2D': True, 'MaxPooling2D': True, 'AveragePooling2D': True, 'GlobalMaxPooling2D': True,
                     'LocallyConnected2D': True, 'Add': True, 'Concatenate': True, 'ReLU': True, 'LeakyReLU': True, 'PReLU': True, 'ELU': True,
                     'Softmax': True, 'BatchNormalization': True, 'Dense': True, 'Flatten': True,
                     'DepthwiseConv2D': True, 'GlobalAveragePooling2D': True}
transition['Softmax'] = {'Conv2D': True, 'Dropout': True, 'SpatialDropout2D': True, 'SeparableConv2D': True,
                         'Conv2DTranspose': True, 'UpSampling2D': True, 'MaxPooling2D': True, 'AveragePooling2D': True, 'GlobalMaxPooling2D': True,
                         'LocallyConnected2D': True, 'Add': True, 'Concatenate': True, 'ReLU': True, 'LeakyReLU': True, 'PReLU': True,
                         'ELU': True, 'Softmax': True, 'BatchNormalization': True, 'Dense': True, 'Flatten': True,
                         'DepthwiseConv2D': True, 'GlobalAveragePooling2D': True}
transition['BatchNormalization'] = {'Conv2D': True, 'Dropout': True, 'SpatialDropout2D': True, 'SeparableConv2D': True,
                                    'Conv2DTranspose': True, 'UpSampling2D': True, 'MaxPooling2D': True, 'AveragePooling2D': True,
                                    'GlobalMaxPooling2D': True, 'LocallyConnected2D': True, 'Add': True, 'Concatenate': True, 'ReLU': True,
                                    'LeakyReLU': True, 'PReLU': True, 'ELU': True, 'Softmax': True, 'BatchNormalization': False,
                                    'Dense': True, 'Flatten': True, 'DepthwiseConv2D': True, 'GlobalAveragePooling2D': True}
transition['Dense'] = {'Conv2D': False, 'Dropout': True, 'SpatialDropout2D': False, 'SeparableConv2D': False,
                       'Conv2DTranspose': False, 'UpSampling2D': False, 'MaxPooling2D': False, 'AveragePooling2D': True, 'GlobalMaxPooling2D': False,
                       'LocallyConnected2D': False, 'Add': True, 'Concatenate': True, 'ReLU': True, 'LeakyReLU': True, 'PReLU': True,
                       'ELU': True, 'Softmax': True, 'BatchNormalization': True, 'Dense': True, 'Flatten': False,
                       'DepthwiseConv2D': False, 'GlobalAveragePooling2D': False}
transition['Flatten'] = {'Conv2D': False, 'Dropout': True, 'SpatialDropout2D': False, 'SeparableConv2D': False,
                         'Conv2DTranspose': False, 'UpSampling2D': False, 'MaxPooling2D': False, 'AveragePooling2D': True, 'GlobalMaxPooling2D': False,
                         'LocallyConnected2D': False, 'Add': True, 'Concatenate': True, 'ReLU': True, 'LeakyReLU': True, 'PReLU': True,
                         'ELU': True, 'Softmax': True, 'BatchNormalization': True, 'Dense': True, 'Flatten': False,
                         'DepthwiseConv2D': True, 'GlobalAveragePooling2D': False}


## setting transition matrices according to layers_list
# i = 17
# l1 = layers_list[i]
# dna_l1 = [l1, {k:layer_params[l1][k]() for k in layer_params[l1].keys()}, -1]
# res = "transition['{}'] = {{".format(l1)
# for l2 in layers_list:
#    dna_l2 = [l2, {k:layer_params[l2][k]() for k in layer_params[l2].keys()}, -1]
#    dna = [dna_l1, dna_l2]
#    print(dna)
#    model, error = create_nn(dna, (50,50,1))
#    if error == '':
#        print('ok')
#        res += "'{}':True, ".format(l2)
#    else:
#        print('*** not ok: {}'.format(error))
#        res += "'{}':False, ".format(l2)
# res = res[:-2]
# res += "}"
# print(res)


class Multihead_generator(keras.utils.Sequence):
    """
    duplicate one output to many, for simultaneous nnets
    """
    def __init__(self, generator, population_size):
        self.parent = generator
        self.population_size = population_size

    def __len__(self):
        # return int(np.ceil(len(self.parent.paths) / float(self.parent.batch_size)))
        return len(self.parent)

    def __getitem__(self, idx):
        X, y = self.parent.__getitem__(idx)
        y = [y for _ in range(self.population_size)]
        return X, y

    def on_epoch_end(self):
        return self.parent.on_epoch_end()


class DNA_DB():
    """
    class for storing info about nnet architectures, its names and mean scores
    """
    def __init__(self, max_times=3):
        """
        class constructor
        :param max_times: max number of certain dna can appear for collecting statistics, default 3
        """
        # TODO: (scores, times) -> [times]
        self.dnas = []
        self.names = []
        self.scores = []
        self.times = []
        self.max_times = max_times

    def add(self, dna, name):
        """
        adding dna to db and return its index. if dna already exist in db then existing index returns. if attempt to add
        dna more than max_times times, then -1 returns.
        :param dna:
        :param name: string name
        :return: index of dna in db
        """
        if dna not in self.dnas:
            self.dnas.append(copy.deepcopy(dna))
            self.names.append(copy.copy(name))
            self.scores.append(-np.inf)
            self.times.append(np.nan)
            ret_idx = len(self.dnas) - 1
        else:
            ret_idx = self.dnas.index(dna)
            if self.times[ret_idx] >= self.max_times:
                ret_idx = -1
            # else:
            #     self.times[ret_idx] += 1
        return ret_idx

    def update_score(self, idx, score):
        """
        update score record for given dna
        :param idx: index in db
        :param score: score to store
        :return:
        """
        if np.isnan(self.times[idx]):
            self.scores[idx] = score
            self.times[idx] = 1
        else:
            self.scores[idx] = (self.scores[idx] * self.times[idx] + score) / (self.times[idx] + 1)
            self.times[idx] += 1

    def get_dna(self, idx):
        return self.dnas[idx]

    def get_name(self, idx):
        return self.names[idx]

    def get_score(self, idx):
        return self.scores[idx]

    def get_times(self, idx):
        return self.times[idx]

    def get_scoreboard(self, n=100):
        """
        get dna indexes ordered by highscores, from better to worse
        :param n: n best does matter, default 100
        :return:
        """
        scoreboard = list(reversed(np.argsort(self.scores)))
        if len(scoreboard) > n:
            scoreboard = scoreboard[:n]
        return scoreboard

    def get_median_score(self, n=100):
        """
        calculates the median score
        :param n: n best does matter, default 100
        :return:
        """
        scoreboard = self.get_scoreboard(n)
        scores = [self.scores[scoreboard[i]] for i in range(len(scoreboard))]
        return np.median(scores)


class Sansara():
    def __init__(self, input_shape, output_shape, population_size, fertile_size=100,  base_dna=None,
                 crossed_part=0.5, epochs=1, params_limit=1000000, tag='project'):
        """
        class constructor
        :param input_shape: shape of input data
        :param output_shape: shape of labels
        :param population_size: itself
        :param fertile_size: number of best dnas from scoreboard to make next generation, both crossing over and mutation
        :param base_dna: dna for start with
        :param crossed_part: part of whole population that are crossovered children
        :param epochs: int number of epochs for evaluate one generation
        :param params_limit: int limit number of nnet parameters for one dna
        :param tag: string to distinct different projects logs and temp savings
        """
        self.input_shape = input_shape
        self.output_shape = tuple([None] + list(output_shape))
        self.population_size = population_size
        self.fertile_size = fertile_size
        self.base_dna = base_dna
        self.crossed_size = int(np.ceil(crossed_part * self.population_size))
        self.epochs = epochs
        self.params_limit = params_limit
        self.dnas = DNA_DB()
        self.turn = 0
        self.genome = []  # list of dnas in current population, for training in parallel
        self.tag = tag
        self.log_filename = '{}.log'.format(self.tag)
        self.vars_filename = '{}_last.pkl'.format(self.tag)

        if not self.state_vars('load'):
            self.create_first_gen()

    def inherit_probs(self, n):
        """
        calculate inheritance probabilities for ordered scoreboard of DNA
        :param n: population size
        :return: inherit_probs: list of probabilities from most successful to least one
        """
        inherit_probs = [p ** 2 for p in np.linspace(1, 0, num=n)]
        # inherit_probs = [np.sqrt(p) for p in np.linspace(1, 0, num=n)]
        inherit_probs /= np.sum(inherit_probs)
        return inherit_probs

    def print_log(self, msg, end='\n'):
        """
        print message to logfile and to console
        :param msg:
        :param end:
        :return:
        """
        with open(self.log_filename, 'a') as outfile:
            print(msg, end=end, file=outfile)
        print(msg, end=end, flush=True)

    def create_nn(self, dna):
        """
        trying to create neural net for given DNA
        :return: model: keras model,
        :return: error: string with error text or empty string if none
        """
        error = ''
        model = []
        in_layer = keras.Input(shape=self.input_shape, name='in_layer')
        model_layers = []
        model_layers.append(in_layer)
        for gene in dna:
            layer_name, params, connect_to = gene
            if layer_name in ['Add', 'Concatenate']:
                layer_call = 'keras.layers.' + layer_name + '()'
                connect2 = params['connect2']
                try:
                    mid_layer = eval(layer_call)([model_layers[connect_to], model_layers[connect2]])
                except (IndexError, ValueError) as e:
                    error = e
                    break
            else:
                params_str = ''
                for key in params.keys():
                    params_str += '{}={},'.format(key, params[key])
                params_str = params_str[:-1]
                layer_call = 'keras.layers.{}({})'.format(layer_name, params_str)
                l = lambda x: eval(layer_call)(x)
                try:
                    mid_layer = l(model_layers[connect_to])
                except (tf.errors.InvalidArgumentError, ValueError) as e:
                    error = e
                    break
            model_layers.append(mid_layer)

        if error == '':
            model = keras.Model(inputs=in_layer, outputs=model_layers[-1])
            if model.output_shape != self.output_shape:
                error = '*** wrong output_shape (expected {}, got {})'.format(self.output_shape, model.output_shape)

        return model, error

    def create_pop_nn(self):
        """
        create neural net for whole population with one shared input and some duplicated outputs for parallel training
        :return: model_pop: keras model
        """
        self.print_log('creating population network.. ', end='')
        in_layer = keras.Input(shape=self.input_shape, name='in_layer')
        model_layers = []
        model_outputs = []
        model_layers.append(in_layer)
        for idx_dna, being_counter in zip(self.genome, range(self.population_size)):

            cur_mod_layer_num = 0
            dna = copy.deepcopy(self.dnas.get_dna(idx_dna))
            name = self.dnas.get_name(idx_dna)
            dna[-1][1]['name'] = name[:-1] + '_{}'.format(being_counter) + name[-1:]
            for gene, i in zip(dna, range(len(dna))):
                layer_name, params, connect_to = gene
                if cur_mod_layer_num == 0:
                    connect_to = 0

                if layer_name in ['Add', 'Concatenate']:
                    layer_call = 'keras.layers.' + layer_name + '()'
                    connect2 = params['connect2']
                    if connect2 + i == -1:  # if adding from an input layer
                        mid_layer = eval(layer_call)([model_layers[connect_to], model_layers[0]])
                    else:
                        mid_layer = eval(layer_call)([model_layers[connect_to], model_layers[connect2]])
                else:
                    params_str = ''
                    for key in params.keys():
                        params_str += '{}={},'.format(key, params[key])
                    params_str = params_str[:-1]
                    layer_call = 'keras.layers.{}({})'.format(layer_name, params_str)
                    l = lambda x: eval(layer_call)(x)
                    mid_layer = l(model_layers[connect_to])

                model_layers.append(mid_layer)
                cur_mod_layer_num += 1

            model_outputs.append(model_layers[-1])

        model_pop = keras.Model(inputs=in_layer, outputs=model_outputs)
        self.print_log('done')

        return model_pop

    def mutate_dna(self, idx_dna):
        """
        apply mutation to given DNA
        :param idx_dna: dna index in db
        :return: new_dna:
        """
        valid = False
        fails_count = 0
        while not valid:
            new_dna = copy.deepcopy(self.dnas.get_dna(idx_dna))

            action = np.random.choice(['insert', 'remove', 'change'])
            if action == 'insert':
                idx = np.random.randint(len(new_dna) - 1)  # last layer cannot be changed
                l1 = new_dna[idx][0]
    #            idx = np.random.randint(len(new_dna)+1)  # last layer can be changed
    #            if idx == 0:
    #                l1 = 'Input'
    #            else:
    #                l1 = new_dna[idx-1][0]
                l2_list = [k for k in transition[l1].keys() if transition[l1][k]]  # select only compatible layers
                l2 = np.random.choice(l2_list)
                new_gene = [l2, {k:layer_params[l2][k]() for k in layer_params[l2].keys()}, -1]
                new_dna.insert(idx + 1, new_gene)

                for i in range(len(new_dna)):  # keep far connections
                    if i != idx + 1:  # skip check for new inserted gene
                        if new_dna[i][2] != -1:
                            connect1_idx = new_dna[i][2] + i
                            connect1_idx_new = connect1_idx - 1
                            if connect1_idx_new < idx < i:  # if removed gene was between layer and it connection point
                                new_dna[i][2] = connect1_idx_new - i

                        if new_dna[i][0] in ['Add', 'Concatenate']:
                            connect2_idx = new_dna[i][1]['connect2'] + i
                            connect2_idx_new = connect2_idx - 1
                            if connect2_idx_new < idx < i:  # if new gene inserted between add/concatenate layer and it connection point
                                new_dna[i][1]['connect2'] = connect2_idx_new - i

                self.print_log('i{}'.format(new_gene[0]), end='')
            elif action == 'remove':
                idx = np.random.randint(len(new_dna)-1)  # last layer cannot be changed
    #            idx = np.random.randint(len(new_dna))  # last layer can be changed
                removed_gene = new_dna[idx]
                new_dna.pop(idx)

                for i in range(len(new_dna)):  # keep far connections
                    if new_dna[i][2] != -1:
                        connect1_idx = new_dna[i][2] + i
                        connect1_idx_new = connect1_idx + 1
                        if connect1_idx_new < idx < i:  # if removed gene was between layer and it connection point
                            new_dna[i][2] = connect1_idx_new - i

                    if new_dna[i][0] in ['Add', 'Concatenate']:
                        connect2_idx = new_dna[i][1]['connect2'] + i
                        connect2_idx_new = connect2_idx + 1
                        if connect2_idx_new < idx < i:  # if removed gene was between add/concatenate layer and it connection point
                            new_dna[i][1]['connect2'] = connect2_idx_new - i

                self.print_log('r{}'.format(removed_gene[0]), end='')
            else:  # change
                idx = np.random.randint(len(new_dna)-1)  # last layer cannot be changed
    #            idx = np.random.randint(len(new_dna))  # last layer can be changed
                if idx == 0:
                    l1 = 'Input'
                else:
                    l1 = new_dna[idx-1][0]
                l2_list = [k for k in transition[l1].keys() if transition[l1][k]]  # select only compatible layers
                l2 = np.random.choice(l2_list)
                changed_gene = new_dna[idx][0]
                if l2 == new_dna[idx][0]:
                    params_list = [k for k in layer_params[l2].keys()]
                    if params_list == []:
                        continue  # if the same parameterless layer has been chosen - then trying to make another mutation
                    params = new_dna[idx][1]  # otherwise mutate one of the parameters
                    param_to_change = np.random.choice(params_list)
                    params[param_to_change] = layer_params[l2][param_to_change]()
                    new_dna[idx][1] = params
                else:
                    new_dna[idx] = [l2, {k:layer_params[l2][k]() for k in layer_params[l2].keys()}, -1]
                self.print_log('c{}'.format(changed_gene),end='')

            m, error = self.create_nn(new_dna)
            if error == '' and m.count_params() <= self.params_limit:
                valid = True
                self.print_log('.', end='')
            else:
                self.print_log('x', end='')
            fails_count += 1

            if fails_count >= 10:
                gc.collect()
                fails_count = 0
                self.print_log('g',end='')
        return new_dna

    def crossing_over(self, idx_dna1, idx_dna2):
        """
        crossing operation for two dnas: get two children
        :param idx_dna1: index of first dna
        :param idx_dna2: index of second dna
        :return child1: dna of first child
        :return child2: dna of second child
        """
        dna1 = copy.deepcopy(self.dnas.get_dna(idx_dna1))
        dna2 = copy.deepcopy(self.dnas.get_dna(idx_dna2))
        valid = False
        while not valid:
            cut_point = np.random.randint(np.min([len(dna1), len(dna2)]))  # point of crossing over
            # cut_point1 = np.random.randint(len(dna1))  # point 1 of crossing over
            # cut_point2 = np.random.randint(len(dna2))  # point 2 of crossing over
            child1 = dna1[0:cut_point].copy() + dna2[cut_point:].copy()
            # child1 = dna1[0:cut_point1].copy() + dna2[cut_point2:].copy()
            m, e = self.create_nn(child1)
            if e == '' and m.count_params() <= self.params_limit:
                child2 = dna2[0:cut_point].copy() + dna1[cut_point:].copy()
                # child2 = dna2[0:cut_point2].copy() + dna1[cut_point1:].copy()
                m, e = self.create_nn(child2)
                if e == '' and m.count_params() <= self.params_limit:
                    valid = True
                    self.print_log('..', end='')
                else:
                    self.print_log('x', end='')
            else:
                self.print_log('x', end='')

        return copy.deepcopy(child1), copy.deepcopy(child2)

    def create_first_gen(self):
        """
        creating list of first generation based on base_dna
        :return:
        """
        if self.base_dna == None:
            pass  # TODO: create first generation with base_dna == None
        else:
            self.print_log('creating first generation from base_dna..')
            idx_dna = self.dnas.add(self.base_dna, "'base_dna'")
            self.genome.append(idx_dna)

            # TODO: make branch names going consecutively
            # i = 0
            for i in range(self.population_size - 1):
            # while len(self.genome) < self.population_size:
                idx_dna_new = -1
                while idx_dna_new < 0:
                    new_dna = copy.deepcopy(self.mutate_dna(0))  # self.base_dna
                    idx_dna_new = self.dnas.add(new_dna, "'g1m{}'".format(i))

                self.genome.append(idx_dna_new)

            self.print_log('   done\ngenome of {} dnas has been generated'.format(len(self.genome)))

    def create_next_gen(self):
        """
        create new generation based on given genome. this include survived part, crossovered part and mutated part
        :return:
        """
        self.print_log('creating next generation based on leaderboard: ')
        genome_new = []
        self.print_log('\nsurvived: ', end='')
        for i in range(self.survived_size):  # survived DNAs
            # genome_new.append(copy.deepcopy(self.genome[self.leaderboard[i]]))
            genome_new.append(self.genome[self.leaderboard[i]])
            self.print_log('.', end='')

        self.print_log('\ncrossovered: ', end='')
        i = 0
        while len(genome_new) < self.survived_size + self.crossed_size:  # creating crossovered
            idx_dna1 = self.genome[np.random.choice(self.leaderboard, p=self.inherit_probs)]
            idx_dna2 = self.genome[np.random.choice(self.leaderboard, p=self.inherit_probs)]
            child1, child2 = self.crossing_over(idx_dna1, idx_dna2)

            idx_child1 = self.dnas.add(child1, "'g{}c{}'".format(self.turn, i))
            if idx_child1 >= 0:
                genome_new.append(idx_child1)
                i += 1
            else:
                self.print_log('-', end='')

            idx_child2 = self.dnas.add(child2, "'g{}c{}'".format(self.turn, i))
            if idx_child2 >= 0:
                genome_new.append(idx_child2)
                i += 1
            else:
                self.print_log('-', end='')

        while len(genome_new) > self.survived_size + self.crossed_size:
            genome_new.pop()
            self.print_log('*', end='')

        self.print_log('\nmutants: ', end='')
        for i in range(self.population_size - self.survived_size - self.crossed_size):  # creating mutants
            idx_dna_new = -1
            while idx_dna_new < 0:
                new_dna = copy.deepcopy(self.mutate_dna(self.genome[np.random.choice(self.leaderboard, p=self.inherit_probs)]))
                idx_dna_new = self.dnas.add(new_dna, "'g{}m{}'".format(self.turn, i))

            genome_new.append(idx_dna_new)
        self.print_log('')
        self.genome = genome_new

    def create_next_gen_global(self):
        """
        create new generation based on best part of all-time scores
        :return:
        """
        genome_new = []
        scoreboard = self.dnas.get_scoreboard(self.fertile_size)

        inherit_probs = self.inherit_probs(len(scoreboard))  # picking probabilities for crossing over and mutations

        self.print_log('\nmedian score for top-{}: {:.6f}'.format(len(scoreboard), self.dnas.get_median_score(self.fertile_size)))

        self.print_log('all-time high mean score: {} = {:.6f}, {} trial(s) out of {}'.format(self.dnas.get_name(scoreboard[0]),
                       self.dnas.get_score(scoreboard[0]), self.dnas.get_times(scoreboard[0]), self.dnas.max_times))

        # print best dna so far
        best_dna = self.dnas.get_dna(scoreboard[0])
        for layer in best_dna:
            self.print_log('{}'.format(layer))

        # if not enough statistics for all-time best dna then try it again
        if self.dnas.get_times(scoreboard[0]) < self.dnas.max_times:
            genome_new.append(scoreboard[0])

        self.print_log('\ncreating next generation based on global scoreboard')
        self.print_log('crossovered: ', end='')
        i = 0
        while len(genome_new) < self.crossed_size:  # creating crossovered
            idx_dna1 = np.random.choice(scoreboard, p=inherit_probs)
            idx_dna2 = np.random.choice(scoreboard, p=inherit_probs)
            child1, child2 = self.crossing_over(idx_dna1, idx_dna2)

            idx_child1 = self.dnas.add(child1, "'g{}c{}'".format(self.turn, i))
            if idx_child1 >= 0:
                genome_new.append(idx_child1)
                i += 1
            else:
                self.print_log('-', end='')

            idx_child2 = self.dnas.add(child2, "'g{}c{}'".format(self.turn, i))
            if idx_child2 >= 0:
                genome_new.append(idx_child2)
                i += 1
            else:
                self.print_log('-', end='')

        while len(genome_new) > self.crossed_size:
            genome_new.pop()
            self.print_log('*', end='')

        self.print_log('\nmutants: ', end='')
        for i in range(self.population_size - self.crossed_size):  # creating mutants
            idx_dna_new = -1
            while idx_dna_new < 0:
                new_dna = copy.deepcopy(self.mutate_dna(np.random.choice(scoreboard, p=inherit_probs)))
                idx_dna_new = self.dnas.add(new_dna, "'g{}m{}'".format(self.turn, i))

            genome_new.append(idx_dna_new)
        self.print_log('')
        self.genome = genome_new

    def state_vars(self, action):
        """
        load or save current state variables
        :param action: 'load' or 'save'
        :return: retcode: True in case of success, False otherwise
        """
        retcode = False
        if action == 'load':
            self.print_log('\n\ninput shape: {}'.format(self.input_shape))
            self.print_log('output shape: {}'.format(self.output_shape))
            self.print_log('population size = {}'.format(self.population_size))
            self.print_log('fertile size = {}'.format(self.fertile_size))
            self.print_log('crossovered size = {}'.format(self.crossed_size))
            self.print_log('epochs for one generation: {}'.format(self.epochs))
            self.print_log('limited number of parameters for one item: {}'.format(self.params_limit))

            if os.path.isfile(self.vars_filename):
                self.print_log('loading previous genomes and scores from {}'.format(self.vars_filename))
                fileload = open(self.vars_filename, 'rb')
                tmp = pickle.load(fileload)
                # self.dnas = tmp['dnas']
                self.dnas.dnas = tmp['dnas'].dnas
                self.dnas.names = tmp['dnas'].names
                self.dnas.scores = tmp['dnas'].scores
                self.dnas.times = tmp['dnas'].times
                self.turn = tmp['turn']
                fileload.close()
                retcode = True
            else:
                self.print_log('no file {} found, nothing has been loaded'.format(self.vars_filename))

        elif action == 'save':
            self.print_log('saving genomes and scores to {}'.format(self.vars_filename))
            filesave = open(self.vars_filename, 'wb')
            pickle.dump({'dnas': self.dnas, 'turn': self.turn}, filesave, -1)
            filesave.close()
            retcode = True
        else:
            self.print_log('*** ERROR: unknown action "{}". Use either "load" or "save"'.format(action))

        return retcode

    def main_cycle(self, generations_num, x_train, y_train=None, val_split=0.1, x_val=None, batch_size=16, optimizer='adam',
                   loss='categorical_crossentropy', metrics=['accuracy']):
        """
        sansara's main cycle
        :param generations_num: int number of generations to evolve
        :param x_train: train dataset as numpy.ndarray, keras generator otherwise
        :param y_train: train labels as numpy.ndarray, keras generator otherwise
        :param val_split: aka keras validation_split
        :param x_val: validation generator (if present)
        :param batch_size: itself
        :param optimizer: itself
        :param loss: itself
        :param metrics: itself
        :return:
        """
        if str(type(x_train)) != '<class \'numpy.ndarray\'>':  # if x_train is generator
            train_gen = Multihead_generator(x_train, self.population_size)
            val_gen = Multihead_generator(x_val, self.population_size)

        for self.turn in range(self.turn + 1, generations_num + 1):
            self.print_log('\nTURN {}'.format(self.turn))

            start_time = time.time()

            if not np.isnan(self.dnas.get_times(0)):  # if not first turn
                # self.create_next_gen()
                self.create_next_gen_global()
            model_pop = self.create_pop_nn()
            # self.print_log('converting model to multi gpu.. ', end='')
            # model_pop = multi_gpu_model(model_pop, gpus=2)
            # self.print_log('done')

            model_pop.compile(optimizer=optimizer, loss=loss, metrics=metrics)
            self.print_log('\nwhole population model have {} parameters'.format(model_pop.count_params()))
            self.print_log('fitting population..')

            if str(type(x_train)) == '<class \'numpy.ndarray\'>':
                history = model_pop.fit(x=x_train, y=[y_train for _ in range(self.population_size)],
                                        validation_split=val_split, batch_size=batch_size, epochs=self.epochs, verbose=2)
            else:
                # history = model_pop.fit_generator(train_pop_gen, epochs=self.epochs, validation_data=val_pop_gen, shuffle=True, verbose=1)
                # history = model_pop.fit_generator(generator=train_gen, steps_per_epoch=x_train.__len__(), epochs=self.epochs,
                #           validation_data=val_gen, validation_steps=x_val.__len__()  , shuffle=True, verbose=1)
                history = model_pop.fit_generator(generator=train_gen, epochs=self.epochs, validation_data=val_gen,
                                                  shuffle=True, verbose=2)

            # scoring childs by last val_f1  # (loss + validation loss)
            for child, idx_dna in zip(model_pop.output_names, self.genome):
                # self.childs_scores['score'].append(history.history[child + '_loss'][-1] + history.history['val_' + child + '_loss'][-1])
                score = history.history['val_' + child + '_acc'][-1]  # get val_acc from last epoch
                self.dnas.update_score(idx_dna, score)

            self.print_log('\nDNAs in database: {}'.format(len(self.dnas.dnas)))
            self.state_vars('save')
            gc.collect()

            if self.turn == 1:  # print performance of base_dna if first turn
                self.print_log('base dna performance: {} = {:.6f}'.format(self.dnas.get_name(0), self.dnas.get_score(0)))

            self.print_log('\n----------- execution time: {} secs -------------'.format(np.int(time.time() - start_time)))

        scoreboard = self.dnas.get_scoreboard(self.fertile_size)
        self.print_log('all-time high mean score: {} = {:.6f}, {} trial(s) out of {}'.format(self.dnas.get_name(scoreboard[0]),
                       self.dnas.get_score(scoreboard[0]), self.dnas.get_times(scoreboard[0]), self.dnas.max_times))

        # print best dna so far
        best_dna = self.dnas.get_dna(scoreboard[0])
        for layer in best_dna:
            self.print_log('{}'.format(layer))
