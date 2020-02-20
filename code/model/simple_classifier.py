import tensorflow as tf
from code.model.environment import env
from code.options import read_options
import logging
import json
import datetime
import os
import uuid
from pprint import pprint
from code.model.trainer import create_permutations
import numpy as np
logger = logging.getLogger()

class SimpleClassifier():
    """
    Simple classifier for fact prediction consisting of a single dense layer. Uses the query relation and object for prediction"
    """

    def __init__(self, params):

        self.batch_size = params['batch_size']
        self.num_rollouts = params['num_rollouts']
        self.action_vocab_size = len(params['relation_vocab'])
        self.entity_vocab_size = len(params['entity_vocab'])
        self.embedding_size = params['embedding_size']
        self.train_env = env(params,'train')
        self.test_env = env(params,'test')
        self.eval_every = params['eval_every']
        self.learning_rate = params['learning_rate_judge']
        self.total_iteration = params['total_iterations']
        self.optimizer = tf.train.AdamOptimizer(self.learning_rate)

    def construct_graph(self):
        logger.info("CONSTRUCTING GRAPH")
        with tf.variable_scope("relation"):
            self.query_relation = tf.placeholder(tf.int32, [None], name="query_relation")
            self.action_embedding_placeholder = tf.placeholder(tf.float32,
                                                               [self.action_vocab_size, 2 * self.embedding_size])

            self.relation_lookup_table = tf.get_variable("relation_lookup_table",
                                                         shape=[self.action_vocab_size, 2 * self.embedding_size],
                                                         dtype=tf.float32,
                                                         initializer=tf.contrib.layers.xavier_initializer(),
                                                         trainable=True)
            self.relation_embedding_init = self.relation_lookup_table.assign(self.action_embedding_placeholder)

        with tf.variable_scope("object"):
            self.query_object = tf.placeholder(tf.int32, [None], name='query_object')
            self.entity_embedding_placeholder = tf.placeholder(tf.float32,
                                                               [self.entity_vocab_size, 2 * self.embedding_size])
            self.entity_lookup_table = tf.get_variable("entity_lookup_table",
                                                       shape=[self.entity_vocab_size, 2 * self.embedding_size],
                                                       dtype=tf.float32,
                                                       initializer=tf.contrib.layers.xavier_initializer(),
                                                       trainable=True)
            self.entity_embedding_init = self.entity_lookup_table.assign(self.entity_embedding_placeholder)

        self.labels = tf.placeholder(tf.float32, [None,1],name="labels")

        self.logits_judge = self.classify(self.query_relation,self.query_object)
        self.loss_judge = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels= self.labels, logits= self.logits_judge))  # tf.reduce_mean(tf.square(self.logits_judge - self.labels))
        self.minimize_opt = self.optimizer.minimize(self.loss_judge)
        logger.info("GRAPH DONE")

    def classify(self,query_relation, query_object):
        self.query_relation_embedding = tf.nn.embedding_lookup(self.relation_lookup_table, query_relation)
        self.query_object_embedding = tf.nn.embedding_lookup(self.entity_lookup_table, query_object)
        query_concat = tf.concat([self.query_relation_embedding, self.query_object_embedding],axis=-1)
        output = tf.layers.dense(query_concat,1,name="classyfier")
        return output


    def predict(self,sess):
        pass
    def train(self,sess):
        counter = 0
        feed_dict = dict()
        for episode in self.train_env.get_episodes():
            query_relation = episode.get_query_relation()
            query_object = episode.get_query_objects()
            label = episode.get_labels()

            feed_dict[self.query_relation] = query_relation
            feed_dict[self.query_object] = query_object
            feed_dict[self.labels] = label

            logits_judge, _ = sess.run([self.logits_judge,self.minimize_opt],feed_dict=feed_dict)
            predictions = logits_judge > 0
            acc = np.mean(predictions == label)
            print(acc)

            counter += 1
            if counter >= self.total_iteration:
                break

        self.test(sess)

    def test(self,sess):
        feed_dict = dict()
        total_acc = 0
        total_examples = 0
        for episode in self.test_env.get_episodes():
            query_relation = episode.get_query_relation()
            query_object = episode.get_query_objects()
            label = episode.get_labels()
            temp_batch_size = episode.no_examples

            feed_dict[self.query_relation] = query_relation
            feed_dict[self.query_object] = query_object
            feed_dict[self.labels] = label
            logits_judge = sess.run(self.logits_judge,feed_dict=feed_dict)

            predictions = logits_judge > 0

            total_acc += np.sum(predictions == label)
            total_examples += temp_batch_size

        logger.info("Acc_Test === {}".format(total_acc / total_examples))


def main():
    options = read_options()
    # Set logging
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter('%(asctime)s: [ %(message)s ]',
                            '%m/%d/%Y %I:%M:%S %p')
    console = logging.StreamHandler()
    console.setFormatter(fmt)
    logfile = None
    logger.addHandler(console)
    # read the vocab files, it will be used by many classes hence global scope
    logger.info('reading vocab files...')
    relation_vocab = json.load(open(options['vocab_dir'] + '/relation_vocab.json'))
    entity_vocab = json.load(open(options['vocab_dir'] + '/entity_vocab.json'))
    logger.info('Reading mid to name map')
    mid_to_word = {}
    # with open('/iesl/canvas/rajarshi/data/RL-Path-RNN/FB15k-237/fb15k_names', 'r') as f:
    #     mid_to_word = json.load(f)
    logger.info('Done..')
    logger.info('Total number of entities {}'.format(len(entity_vocab)))
    logger.info('Total number of relations {}'.format(len(relation_vocab)))
    save_path = ''
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = False
    config.log_device_placement = False

    best_permutation = None
    best_acc = 0
    for permutation in create_permutations(options):
        current_time = datetime.datetime.now()
        current_time = current_time.strftime('%y_%b_%d__%H_%M_%S')
        permutation['output_dir'] = options['base_output_dir'] + '/' + str(current_time) + '__' + str(uuid.uuid4())[
                                                                                                  :4] + '_' + str(
            permutation['path_length']) + '_' + str(permutation['beta']) + '_' + str(
            permutation['test_rollouts']) + '_' + str(
            permutation['Lambda'])

        permutation['model_dir'] = permutation['output_dir'] + '/' + 'model/'

        permutation['load_model'] = (permutation['load_model'] == 1)

        ##Logger##
        permutation['path_logger_file'] = permutation['output_dir']
        permutation['log_file_name'] = permutation['output_dir'] + '/log.txt'
        os.makedirs(permutation['output_dir'])
        os.mkdir(permutation['model_dir'])
        with open(permutation['output_dir'] + '/config.txt', 'w') as out:
            pprint(permutation, stream=out)

        # print and return
        maxLen = max([len(ii) for ii in permutation.keys()])
        fmtString = '\t%' + str(maxLen) + 's : %s'
        print('Arguments:')
        for keyPair in sorted(permutation.items()): print(fmtString % keyPair)
        logger.removeHandler(logfile)
        logfile = logging.FileHandler(permutation['log_file_name'], 'w')
        logfile.setFormatter(fmt)
        logger.addHandler(logfile)
        permutation['relation_vocab'] = relation_vocab
        permutation['entity_vocab'] = entity_vocab


        judge = SimpleClassifier(permutation)
        judge.construct_graph()
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            judge.train(sess)


if __name__ == "__main__":
    main()

