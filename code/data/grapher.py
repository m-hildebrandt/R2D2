from collections import defaultdict
import logging
import numpy as np
import csv
import os
import json
import random
from functools import reduce

logger = logging.getLogger(__name__)



class RelationEntityGrapher:
    '''
    Class for parsing and querying the knowledge graph. Adapted from https://github.com/shehzaadzd/MINERVA.

    '''


    def __init__(self, data_input_dir, mode, entity_vocab, relation_vocab, mid_to_name, max_num_actions):
        '''
        Initializes grapher.

        :param data_input_dir: String. Path to the dataset's folder.
        :param mode: String. Either 'train', 'test' or 'dev'.
        :param entity_vocab: dict. Dictionary mapping entity names to unique integer ids.
        :param relation_vocab: dict. Dictionary mapping relation names to unique integer ids.
        :param mid_to_name: dict or None. If dict, dictionary mapping the mid codes from FB15k to human readable names.
        Only used for the FB15k dataset and its variants.
        :param max_num_actions: int. Maximum branch factor of the knowledge graph.
        '''

        self.ePAD = entity_vocab['PAD']
        self.rPAD = relation_vocab['PAD']
        self.QUERY_SUBJECT_NAME = 'QUERY_SUBJECT'
        self.query_subject = entity_vocab[self.QUERY_SUBJECT_NAME]
        self.triple_store = data_input_dir + '/' + 'graph.txt'
        self.dev_triple_store = data_input_dir + '/' + 'dev.txt'
        self.test_triple_store = data_input_dir + '/' + 'test.txt'
        self.relation_vocab = relation_vocab
        self.entity_vocab = entity_vocab
        self.mode = mode
        if mid_to_name is None:
            self.is_FB15K = False
        else:
            self.is_FB15K  = True
            self.mid_to_name = mid_to_name
            self.rev_mid_to_name = dict([(v,k) for k,v in self.mid_to_name.items()])
        self.store = defaultdict(list)
        self.array_store = np.ones((len(entity_vocab), max_num_actions, 2), dtype=np.dtype('int32'))
        self.array_store[:, :, 0] *= self.ePAD
        self.array_store[:, :, 1] *= self.rPAD
        self.rev_relation_vocab = dict([(v, k) for k, v in relation_vocab.items()])
        self.rev_entity_vocab = dict([(v, k) for k, v in entity_vocab.items()])
        self.create_graph()
        logger.info("KG constructed")

    def get_placeholder_subject(self):
        '''
        Getter for the id of the 'QUERY_NAME' placeholder
        :return: int. Id of the 'QUERY_NAME' placeholder.
        '''

        return self.query_subject

    def get_placeholder_string(self):
        '''
        Getter for the string placeholder 'QUERY_NAME'
        :return: str. 'QUERY_NAME'.
        '''
        return self.QUERY_SUBJECT_NAME

    def get_fact_string(self, subject, predicate, object):
        '''
        Maps a triple of ids to a triple of names so it is readable.

        :param subject: int. Id of the triple's subject.
        :param predicate: int. Id of the triple's relation.
        :param object: int. Id of the triple's object.
        :return: String representing the triple.
        '''

        if self.is_FB15K:
            subject = self.rev_entity_vocab[subject]
            if subject in self.mid_to_name:
                subject = self.mid_to_name[subject]
            object = self.rev_entity_vocab[object]
            if object in self.mid_to_name:
                object = self.mid_to_name[object]
            return subject + '\n' + self.rev_relation_vocab[predicate] + '\n' + object
        else:
            return self.rev_entity_vocab[subject] + '\n' + self.rev_relation_vocab[predicate] + '\n' + self.rev_entity_vocab[object]

    def get_entity_string(self, entity):
        '''
        Maps an id subject to its readable name.

        :param subject:
        :return:
        '''
        if self.is_FB15K:
            str_entity = self.rev_entity_vocab[entity]
            if str_entity in self.mid_to_name:
                str_entity = self.mid_to_name[str_entity]
            return str_entity
        else:
            return self.rev_entity_vocab[entity]

    def get_relation_string(self, relation):
        return self.rev_relation_vocab[relation]


    def create_graph(self):
        '''
        Parses knowledge graph into a numpy array.

        Creates self.array_store, a numpy array with shape [num_subjects, max_num_actions, 2] that represents the knowledge graph.
        The first dimension corresponds to entities in the KG. The second is all available actions. First value of the
        third dimension is the object of the triple. Second value, the relation.

        E.g. self.array_store[3,10,0] == 34 and self.array_store[3,10,1] == 4
         means that the triple (3,4,34) belongs to the knowledge graph and is the tenth triple stored for subject '3'.

        The first action for every subject is the 'NO_OP' triple.

        :return: None.

        '''

        with open(self.triple_store) as triple_file_raw:
            triple_file = csv.reader(triple_file_raw, delimiter='\t')
            for line in triple_file:
                e1 = self.entity_vocab[line[0]]
                r = self.relation_vocab[line[1]]
                e2 = self.entity_vocab[line[2]]
                self.store[e1].append((r, e2))

        for e1 in self.store:
            num_actions = 1
            self.array_store[e1, 0, 1] = self.relation_vocab['NO_OP']
            self.array_store[e1, 0, 0] = e1
            for r, e2 in self.store[e1]:
                if num_actions == self.array_store.shape[1]:
                    break
                self.array_store[e1,num_actions,0] = e2
                self.array_store[e1,num_actions,1] = r
                num_actions += 1
        del self.store
        self.store = None

    def return_next_actions(self, current_entities, start_entities, query_relations, end_entities, labels, all_correct_answers, rollouts):
        '''
        Returns the next possible transitions for an agent navigating the knowledge graph.

        :param current_entities: Numpy array, [batch_size]. Array with the ids of the current entities for the batch.
        :param start_entities: Numpy array, [batch_size]. Array with the ids of the subjects for the batch's queries.
        :param query_relations: Numpy array, [batch_size]. Array with the ids of the relations for the batch's queries.
        :param end_entities: Numpy array, [batch_size]. Array with the ids of the objects for the batch's queries.
        :param labels: Numpy array, [batch_size]. Array with the truth values of hte batch's queries.
        :param all_correct_answers: Dict of lists. Dictionary mapping the a subject-relation pair to a list of
        all correct objects in the knowledge graph.
        :param rollouts: int. Number of rollouts in the batch.
        :return: Numpy array, [batch_size, max_num_actions, 2]. Subset array of self.array_store with only the entries
        corresponding to the current entities and correct answer hidden.
        '''

        ret = self.array_store[current_entities, :, :].copy()

        # During testing of false triples, we don't want the agents to pick the right answer and trivially show it as
        # counter-argument. To avoid this, we utilize the fact that the first entry in the batch during testing
        # corresponds to the right answer.
        correct_pred = query_relations[0]
        correct_obj = end_entities[0]
        for i in range(current_entities.shape[0]):

            if self.mode == 'train':
                correct_pred = query_relations[i]

                if labels[i]:
                    # In Train mode, we want to pick hide either the object if the triple is true...
                    correct_obj = end_entities[i]
                else:
                    # ...or a random triple that that is correct.
                    # Otherwise, the con agent might learn to present the right answer as a counter-argument, which
                    # it can't do during testing, as we hide it. (See above).
                    correct_obj = random.sample(all_correct_answers[i // rollouts], 1)[0]

            if current_entities[i] == start_entities[i]:
                relations = ret[i, :, 1]
                entities = ret[i, :, 0]
                mask = np.logical_and(relations == query_relations[i], entities == end_entities[i])
                ret[i, :, 0][mask] = self.ePAD
                ret[i, :, 1][mask] = self.rPAD

                mask_correct_answer = np.logical_and(relations == correct_pred, entities == correct_obj)
                ret[i, :, 0][mask_correct_answer] = self.ePAD
                ret[i, :, 1][mask_correct_answer] = self.rPAD


        return ret