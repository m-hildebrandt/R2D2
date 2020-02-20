from tqdm import tqdm
import json
import numpy as np
from collections import defaultdict
import csv
import random
import os
import sys


class RelationEntityBatcher():
    '''
    Class for creating and returning the batches for training and testing. Adapted from https://github.com/shehzaadzd/MINERVA.
    '''

    def __init__(self, input_dir, batch_size, entity_vocab, relation_vocab, is_use_fixed_false_facts, num_false_facts=0,
                 mode="train", rounds_sub_training=0):
        '''
        Initializes the batcher.

        :param input_dir: String. Path to the dataset's folder.
        :param batch_size: int. Batch size.
        :param entity_vocab: dict. Dictionary mapping entity names to unique integer ids.
        :param relation_vocab: dict. Dictionary mapping relation names to unique integer ids.
        :param is_use_fixed_false_facts: bool. Flag used to tell whether during testing the batch should use a fixed set
        :param total_iterations:
        of false facts (for fact prediction) or whether the false facts should be computed at run time (for link prediction)
        :param num_false_facts: int. Num of false facts to compute if is_use_fixed_false_facts is set to True.
        :param mode: String. Either 'train', 'test', or 'dev'. Mode the batcher is running in.
        :param rounds_sub_training: Int. Number of rounds for unbiased training batch-generation.
        '''

        self.input_dir = input_dir
        self.input_file = input_dir+'/{0}.txt'.format(mode)
        self.batch_size = batch_size
        self.num_false_facts = num_false_facts
        self.entity_vocab = entity_vocab
        self.relation_vocab = relation_vocab
        self.mode = mode
        self.is_use_fixed_false_facts = is_use_fixed_false_facts
        self.rounds_sub_training = rounds_sub_training #TODO: Test if biased training for the agents changes something.
        self.create_triple_store()


    def create_triple_store(self):
        '''
        Creates a store for the triples used during training/testing and populates dictionaries used for negative triple
        generation.

        :return: None.

        '''

        self.store_relation_objects = defaultdict(set)
        self.store_relation_subjects = defaultdict(set)
        self.store_relation_object_subjects = defaultdict(set)
        self.store_all_correct = defaultdict(set)
        self.store = []
        if self.mode == 'train':
            with open(self.input_dir+'/train.txt') as raw_input_file:
                csv_file = csv.reader(raw_input_file, delimiter = '\t' )
                for line in csv_file:
                    e1 = self.entity_vocab[line[0]]
                    r = self.relation_vocab[line[1]]
                    e2 = self.entity_vocab[line[2]]
                    is_correct = int(line[3]) if len(line) == 4 else 1 #If no information, assume fact is correct.
                    self.store.append([e1,r,e2,is_correct])
                    if is_correct:
                        self.store_all_correct[(e1, r)].add(e2)
                        self.store_relation_objects[r].add(e2)
                        self.store_relation_subjects[r].add(e1)
                        self.store_relation_object_subjects[(r,e2)].add(e1)
        else:
            self.store_false = defaultdict(set)
            with open(self.input_dir+'/{0}.txt'.format(self.mode)) as raw_input_file:
                csv_file = csv.reader(raw_input_file, delimiter = '\t' )
                # We only use fixed false samples for dev and test set.
                if self.is_use_fixed_false_facts:
                    # Save false facts separately. This code block assumes that after the negative samples are produced
                    # by replacing the object of a true triple. Furthermore, it assumes that all corresponding false samples
                    # immediately follow the correct triple.
                    for line in csv_file:
                        e1 = line[0]
                        r = line[1]
                        e2 = line[2]
                        if e1 in self.entity_vocab and e2 in self.entity_vocab:
                            e1 = self.entity_vocab[e1]
                            r = self.relation_vocab[r]
                            e2 = self.entity_vocab[e2]
                            is_correct = int(line[3])
                            if is_correct:
                                correct_e2 = e2
                                self.store.append([e1, r, e2, is_correct])
                            else:
                                self.store_false[(e1,r,correct_e2)].add(e2)
                # Only true facts are considered
                else:
                    for line in csv_file:
                        e1 = line[0]
                        r = line[1]
                        e2 = line[2]
                        if e1 in self.entity_vocab and e2 in self.entity_vocab:
                            e1 = self.entity_vocab[e1]
                            r = self.relation_vocab[r]
                            e2 = self.entity_vocab[e2]
                            assert len(line) == 3 #Make sure the data set and the mode are consistent.
                            is_correct = 1
                            self.store.append([e1, r, e2, is_correct])
                            self.store_relation_subjects[r].add(e1)
                            self.store_relation_object_subjects[(r, e2)].add(e1)

            #To generate the set of plausible but negative samples during testing, we want to consider all available triples.
            fact_files = ['train.txt', 'test.txt', 'dev.txt', 'graph.txt']
            for f in fact_files:
                with open(self.input_dir+'/'+f) as raw_input_file:
                    csv_file = csv.reader(raw_input_file, delimiter='\t')
                    for line in csv_file:
                        e1 = line[0]
                        r = line[1]
                        e2 = line[2]
                        if e1 in self.entity_vocab and e2 in self.entity_vocab:
                            e1 = self.entity_vocab[e1]
                            r = self.relation_vocab[r]
                            e2 = self.entity_vocab[e2]
                            is_correct = int(line[3]) if len(line) == 4 else 1
                            if is_correct:
                                self.store_all_correct[(e1, r)].add(e2)
                                self.store_relation_objects[r].add(e2)

        self.store = np.array(self.store)
        np.random.shuffle(self.store)

    def generate_negative_object(self, subject, relation,used_set):
        '''
        Returns an object's id such that (subject, relation, object) is a negative triple.

        If possible, the object will be different to the ones found in used_set

        :param subject: Subject's id.
        :param relation: Relation's id.
        :param used_set: Set of objects' id to be excluded.
        :return: Int. Object's id producing a false triple with subject and relation.
        '''

        diff_set = self.store_relation_objects[relation].difference(self.store_all_correct[(subject,relation)])\
            .difference(used_set)
        try:
            rand_obj = random.sample(diff_set,1)[0]
        except ValueError:
            diff_set = self.store_relation_objects[relation].difference(self.store_all_correct[(subject,relation)])
            rand_obj = random.sample(diff_set,1)[0]

        return rand_obj


    def generate_negative_subject(self, relation, obj, used_set):
        '''
        Returns a subject's id such that (subject, relation, obj) is a negative triple.

        If possible, the subject will be different to the ones found in used_set.

        :param relation: Relation's id.
        :param obj: Object's id.
        :param used_set: Set of subjects' id to be excluded.
        :return: Int. Subject's id producing a false triple with relation and obj.
        '''

        diff_set = self.store_relation_subjects[relation].difference(self.store_relation_object_subjects[(relation,obj)])\
            .difference(used_set)
        try:
            rand_sub = random.sample(diff_set,1)[0]
        except ValueError:
            diff_set = self.store_relation_subjects[relation].difference(self.store_relation_subjects[(relation,obj)])
            rand_sub = random.sample(diff_set,1)[0]

        return  rand_sub


    def generate_all_false_but_plausible_triples(self, subject, relation):
        '''
        Returns a list of all false but plausible triples for the subject and relation.

        :param subject: Subject's id
        :param relation: Relation's id.
        :return: List. List of false but plausible triples using the subject and relation.
        '''

        set_negatives = self.store_relation_objects[relation].difference(self.store_all_correct[(subject,relation)])
        set_negative_facts = map(lambda object: [subject,relation,object,0],set_negatives)
        return list(set_negative_facts)

    def generate_false_triples(self, batch):
        '''
        Complements a numpy array of size [batch_size, 4] containing correct facts with false_facts-many false triples
        by substituting the subject.

        If possible, the generated false facts will be different from each other. If there is no plausible subject to
        produce a false fact, the true triple is repeated.

        :param batch: batch of correct triples [batch_size,4]
        :return: batch of correct and false triples [batch_size * (false_facts + 1), 4]
        '''
        ret_list = []
        for row in batch:
            ret_list.append(row)
            relation = row[1]
            obj = row[2]
            used_sub = set()
            try:
                for i in range(self.num_false_facts):
                    false_sub = self.generate_negative_subject(relation, obj, used_sub)
                    ret_list.append([false_sub,relation,obj,0])
                    used_sub.add(false_sub)
            except ValueError:
                for i in range(self.num_false_facts):
                    ret_list.append(row)

        return np.array(ret_list)

    def generate_false_triples_biased(self, batch):
        '''
        Complements a numpy array batch [batch_size, 4] containing correct facts with false_facts-many false triples
        by substituting the object.

        If possible, the generated false facts will be different from each other. If there is no plausible object to
        produce a false fact, the true triple is repeated.

        :param batch:  batch of correct triples [batch_size,4]
        :return: batch 0f correct and false triples [batch_size * (false_facts + 1), 4]
        '''
        ret_list = []
        for row in batch:
            ret_list.append(row)
            subject = row[0]
            relation = row[1]
            used_obj = set()
            try:
                for i in range(self.num_false_facts):
                    false_obj = self.generate_negative_object(subject, relation, used_obj)
                    ret_list.append([subject, relation, false_obj,0])
                    used_obj.add(false_obj)
            except ValueError:
                for i in range(self.num_false_facts):
                    ret_list.append(row)

        return np.array(ret_list)


    def yield_next_batch_train(self):
        '''
        Returns a batch for training.

        The batch contains negative samples but no multiple rollouts. Furthermore, the batch is returned as 4 different
        elements:
            - e1: Numpy array [batch_size * (1 + self.num_false_facts)]. The subject of the training samples.
            - r:  Numpy array [batch_size * (1 + self.num_false_facts)]. The relation of the training samples.
            - e2: Numpy array [batch_size * (1 + self.num_false_facts)]. The object of the training samples.
            - is_correct: Numpy array [batch_size * (1 + self.num_false_facts)]. The label of the training samples.
            - all_e2s: List of sets containing all correct objects for each subject-relation pair in the batch.

        Depending on the iter_counter the generated batch is produced in an biased or unbiased manner. The difference is
        whether the subject of the correct triple is replaced (unbiased) or the object (biased)

        :return: Tuple of 4 elements representing the batch.
        '''

        iter_counter = 0
        while True:
            batch_idx = np.random.randint(0, self.store.shape[0], size=self.batch_size)
            batch = self.store[batch_idx, :]
            if iter_counter < self.rounds_sub_training :
                batch = self.generate_false_triples(batch)
            else:
                batch = self.generate_false_triples_biased(batch)



            e1 = batch[:,0]
            r = batch[:, 1]
            e2 = batch[:, 2]
            is_correct = batch[:,3]
            all_e2s = []
            for i in range(e1.shape[0]):
                all_e2s.append(self.store_all_correct[(e1[i], r[i])])

            assert e1.shape[0] == e2.shape[0] == r.shape[0] == is_correct.shape[0] == len(all_e2s)
            iter_counter += 1
            yield e1, r, e2, is_correct, all_e2s

    def yield_next_batch_test(self):
        '''
        Returns a batch for testing.

        The batch contains negative samples but no multiple rollouts. Furthermore, the batch is returned as 4 different
        elements:
            - e1: Numpy array [batch_size * (1 + self.num_false_facts)]. The subject of the training samples.
            - r:  Numpy array [batch_size * (1 + self.num_false_facts)]. The relation of the training samples.
            - e2: Numpy array [batch_size * (1 + self.num_false_facts)]. The object of the training samples.
            - is_correct: Numpy array [batch_size * (1 + self.num_false_facts)]. The label of the training samples.
            - all_e2s: List of sets containing all correct objects for each subject-relation pair in the batch.

        The method keeps yielding testing batchings until the whole set has been transversed once.

        :return: Tuple of 4 elements representing the batch.
        '''

        remaining_triples = self.store.shape[0]
        current_idx = 0
        while True:
            if remaining_triples == current_idx:
                return

            if self.is_use_fixed_false_facts:
                true_batch = self.store[current_idx, :]
                subject = true_batch[0]
                predicate = true_batch[1]
                correct_obj = true_batch[2]
                list_negative_facts = list(map(lambda obj: [subject,predicate,obj,0],self.store_false[(subject,predicate,correct_obj)]))
                batch_list = [true_batch]
                batch_list += list_negative_facts
                batch = np.array(batch_list)
            else:
                true_batch = self.store[current_idx, :]
                subject = true_batch[0]
                predicate = true_batch[1]
                list_negative_facts = self.generate_all_false_but_plausible_triples(subject, predicate)
                batch_list = [true_batch]
                batch_list += list_negative_facts
                batch = np.array(batch_list)


            e1 = batch[:, 0]
            r = batch[:, 1]
            e2 = batch[:, 2]
            is_correct = batch[:,3]
            all_e2s = []
            for i in range(e1.shape[0]):
                all_e2s.append(self.store_all_correct[(e1[i], r[i])])

            assert e1.shape[0] == e2.shape[0] == r.shape[0] == is_correct.shape[0] == len(all_e2s)

            current_idx += 1
            yield e1, r, e2, is_correct, all_e2s
