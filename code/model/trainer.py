from __future__ import absolute_import
from __future__ import division
from tqdm import tqdm
import json
import time
import pickle
import os
import uuid
import datetime
from pprint import pprint
import logging
import numpy as np
from scipy.special import expit as sig
import tensorflow as tf
from code.model.judge import Judge
from code.model.agent import Agent
from code.options import read_options
from code.model.environment import env
from sklearn.model_selection import ParameterGrid
import itertools
import random
import sklearn
import gc

if os.name == 'posix':
    import resource
import sys
from code.model.baseline import ReactiveBaseline
from code.model.debate_printer import  Debate_Printer

logger = logging.getLogger()
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)


class Trainer(object):
    '''
    Central class for R2D2. The trainer is responsible for all components of the model (agent, judge, grapher, etc.) and
    for the training and testing. Adapted from Adapted from the agent class from https://github.com/shehzaadzd/MINERVA.
    '''


    def __init__(self, params, best_metric):
        '''
        Initializes the trainer.

        :param params: Dict. Dictionary with all parameters of the experiment.
        :param best_metric: int. Best metric (either acc or MRR) found during the sweep until that point. Used for
        testing only the best peforming model on the dev test during grid search.
        '''

        # transfer parameters to self
        for key, val in params.items(): setattr(self, key, val);
        self.set_random_seeds(self.seed)
        self.batch_size = self.batch_size *  (1 + self.false_facts_train) * self.num_rollouts
        self.judge = Judge(params)
        self.agent = Agent(params, self.judge)
        self.train_environment = env(params, 'train')
        self.dev_test_environment = env(params, 'dev')
        self.test_test_environment = env(params, 'test')
        self.number_steps = self.path_length * self.number_arguments * 2
        self.best_metric = best_metric

        # optimize
        self.learning_rate_judge_init = params['learning_rate_judge']
        self.learning_rate_judge_ph = tf.placeholder(tf.float32, shape=(), name="learning_rate_judge") #TODO: Dele the placeholder?
        self.baseline_1 = ReactiveBaseline(l=self.Lambda)
        self.baseline_2 = ReactiveBaseline(l=self.Lambda)
        self.optimizer_judge = tf.train.AdamOptimizer(self.learning_rate_judge_ph)
        self.optimizer_agents = tf.train.AdamOptimizer(self.learning_rate_agents)


    def set_random_seeds(self,seed):
        '''
        If provided, sets a random seed.

        Due to Tensorflow's implementation, the code is intrinsically not-deterministic. However, setting the seed helps
        reduce variant and makes the sequence of presented training samples the same across experiments.

        :param seed: int. Seed to set.
        :return: None
        '''
        if not seed is None:
            np.random.seed(seed)
            tf.set_random_seed(seed)
            random.seed(seed)


    def calc_reinforce_loss(self):
        '''
        Calculates the REINFORCE loss.

        :return: Tuple of two tensors, each []. REINFORCE loss for each agent.
        '''

        loss = tf.stack(self.per_example_loss, axis=1)  # [B, T]
        mask = tf.cast(tf.stack(self.which_agent_sequence), tf.bool)
        mask = tf.tile(tf.expand_dims(mask, 0), [loss.get_shape()[0], 1])
        not_mask = tf.logical_not(mask)
        loss_1 = tf.reshape(tf.boolean_mask(loss, not_mask), [loss.get_shape()[0], -1])
        loss_2 = tf.reshape(tf.boolean_mask(loss, mask), [loss.get_shape()[0], -1])

        # multiply with rewards
        if self.custom_baseline:
            final_reward_1 = self.cum_discounted_reward_1
            final_reward_2 = self.cum_discounted_reward_2
        else:
            self.tf_baseline_1 = self.baseline_1.get_baseline_value()
            self.tf_baseline_2 = self.baseline_2.get_baseline_value()

            final_reward_1 = self.cum_discounted_reward_1 - self.tf_baseline_1
            final_reward_2 = self.cum_discounted_reward_2 - self.tf_baseline_2

        reward_mean_1, reward_var_1 = tf.nn.moments(final_reward_1, axes=[0, 1])
        reward_mean_2, reward_var_2 = tf.nn.moments(final_reward_2, axes=[0, 1])

        # Constant added for numerical stability
        reward_std_1 = tf.sqrt(reward_var_1) + 1e-6
        reward_std_2 = tf.sqrt(reward_var_2) + 1e-6

        final_reward_1 = tf.div(final_reward_1 - reward_mean_1, reward_std_1)
        final_reward_2 = tf.div(final_reward_2 - reward_mean_2, reward_std_2)

        loss_1 = tf.multiply(loss_1, final_reward_1)  # [B, T]

        loss_2 = tf.multiply(loss_2, final_reward_2)  # [B, T]

        entropy_policy_1, entropy_policy_2 = self.entropy_reg_loss(self.per_example_logits)
        total_loss_1 = tf.reduce_mean(loss_1) - self.decaying_beta * entropy_policy_1  # scalar

        total_loss_2 = tf.reduce_mean(loss_2) - self.decaying_beta * entropy_policy_2  # scalar

        return total_loss_1, total_loss_2


    def entropy_reg_loss(self, all_logits):
        '''
        Calculates the entropy value of the episode for regularization of the REINFORCE loss.

        :param all_logits: List of tensors, each [Batch_size, max_num_actions]. Tensors with the logits for all available
        actions during the respective timesteps.
        :return: Tuple of two tensors, each []. Entropy regularization values.
        '''

        all_logits = tf.stack(all_logits, axis=2)  # [B, MAX_NUM_ACTIONS, T]
        mask = tf.cast(tf.stack(self.which_agent_sequence), tf.bool)
        mask = tf.tile(tf.expand_dims(mask, 0), [all_logits.get_shape()[1], 1])
        mask = tf.tile(tf.expand_dims(mask, 0), [all_logits.get_shape()[0], 1, 1])
        not_mask = tf.logical_not(mask)
        logits_1 = tf.reshape(tf.boolean_mask(all_logits, not_mask),
                              [all_logits.get_shape()[0], all_logits.get_shape()[1], -1])
        logits_2 = tf.reshape(tf.boolean_mask(all_logits, mask),
                              [all_logits.get_shape()[0], all_logits.get_shape()[1], -1])
        entropy_policy_1 = - tf.reduce_mean(tf.reduce_sum(tf.multiply(tf.exp(logits_1), logits_1), axis=1))  # scalar
        entropy_policy_2 = - tf.reduce_mean(tf.reduce_sum(tf.multiply(tf.exp(logits_2), logits_2), axis=1))  # scalar

        return entropy_policy_1, entropy_policy_2


    def initialize(self, restore=None, sess=None):
        '''
        Constructs the Tensorflow graph for the experiment.

        This method is responsible for creating most of the placeholders used during training and testing and for assigning
        them to the agents and judge. The method is divided in two parts: in the first one, the operations and tensors for
        training are created, in the second one the operations and tensors for testing.

        :param restore: bool. Flag to tell if the weights of the graph should be restore or randomly initialized.
        :param sess: Tf.Session. Session to use for restoring the values if restore is True.
        :return: If restore is True, the initialization operation for the graph. If restore is False, None.
        '''

        logger.info("Creating TF graph...")
        self.which_agent_sequence = []
        self.candidate_relation_sequence = []
        self.candidate_entity_sequence = []
        self.input_path = []

        self.query_subject = tf.placeholder(tf.int32, [None], name="query_subject")
        self.query_relation = tf.placeholder(tf.int32, [None], name="query_relation")
        self.query_object = tf.placeholder(tf.int32, [None], name="query_object")

        self.labels = tf.placeholder(tf.float32, [None, 1], name="labels")
        self.judge.set_labels_placeholder(self.labels)

        self.random_flag = tf.placeholder(tf.bool, [])
        self.range_arr = tf.placeholder(tf.int32, shape=[None, ])
        self.global_step = tf.Variable(0, trainable=False)
        self.decaying_beta = tf.train.exponential_decay(self.beta, self.global_step,
                                                        200, 0.90, staircase=False)
        self.entity_sequence = []

        # to feed in the discounted reward tensor
        self.cum_discounted_reward_1 = tf.placeholder(tf.float32, [None, self.number_steps // 2],
                                                      name="cumulative_discounted_reward")
        self.cum_discounted_reward_2 = tf.placeholder(tf.float32, [None, self.number_steps // 2],
                                                      name="cumulative_discounted_reward")  # Regardless of order, each agent has the same number of steps


        self.judge.set_query_embeddings(self.query_subject, self.query_relation, self.query_object)
        self.agent.set_query_embeddings(self.query_subject, self.query_relation, self.query_object)

        #Populating the placeholder-lists that will then be fed the different time-step dependent values.
        for t in range(self.number_steps):
            which_agent = tf.placeholder(tf.float32, shape=(), name="which_agent_{}".format(t))
            next_possible_relations = tf.placeholder(tf.int32, [None, self.max_num_actions],
                                                     name="next_relations_{}".format(t))
            next_possible_entities = tf.placeholder(tf.int32, [None, self.max_num_actions],
                                                    name="next_entities_{}".format(t))

            input_label_relation = tf.placeholder(tf.int32, [None], name="input_label_relation_{}".format(t))
            start_entities = tf.placeholder(tf.int32, [None, ])
            self.which_agent_sequence.append(which_agent)
            self.input_path.append(input_label_relation)
            self.candidate_relation_sequence.append(next_possible_relations)
            self.candidate_entity_sequence.append(next_possible_entities)
            self.entity_sequence.append(start_entities)

        self.loss_judge, self.final_logits_judge, self.temp_logits_judge, self.per_example_loss, self.per_example_logits, \
        self.action_idx, self.rewards_agents, self.rewards_before_baseline \
            = self.agent(self.which_agent_sequence, self.candidate_relation_sequence, self.candidate_entity_sequence,
                         self.entity_sequence, self.range_arr,
                         self.number_steps, random_flag=self.random_flag)

        self.train_judge = self.bp_judge(self.loss_judge)

        self.loss_1, self.loss_2 = self.calc_reinforce_loss()
        self.train_op_1, self.train_op_2 = self.bp(self.loss_1, self.loss_2)

        # Building the test graph
        self.t = tf.placeholder(tf.int32, shape=[], name="time_step")
        self.input_argument = tf.placeholder(tf.float32, name="partial_argument")
        self.input_hidden_argu_rep= tf.placeholder(tf.float32, shape=[None, self.hidden_size], name="rep_argument")
        self.next_relations = tf.placeholder(tf.int32, shape=[None, self.max_num_actions])
        self.next_entities = tf.placeholder(tf.int32, shape=[None, self.max_num_actions])
        self.which_agent = tf.placeholder(tf.float32, name="which_agent")
        self.current_entities = tf.placeholder(tf.int32, shape=[None, ])
        self.prev_relation = tf.placeholder(tf.int32, [None, ], name="previous_relation")

        self.prev_state_agent_1 = tf.placeholder(tf.float32, self.agent.get_mem_shape(), name="memory_of_agent_1")
        formated_state_agent_1 = self.agent.format_state(self.prev_state_agent_1)

        self.prev_state_agent_2 = tf.placeholder(tf.float32, self.agent.get_mem_shape(), name="memory_of_agent_2")
        formated_state_agent_2 = self.agent.format_state(self.prev_state_agent_2)

        self.output_ = tf.placeholder(shape=[None, self.judge.hidden_size], dtype=tf.float32)
        self.test_loss, \
            test_state_agent_1, test_state_agent_2, self.test_logits, self.test_action_idx, \
            self.chosen_relation = self.agent.step(self.next_relations, self.next_entities,
                                                     formated_state_agent_1, formated_state_agent_2, self.prev_relation, self.current_entities,
                                                     self.range_arr, self.which_agent, random_flag=self.random_flag)

        self.ret_argument = self.judge.extend_argument(self.input_argument,self.t,self.test_action_idx,
                                                       self.next_relations,self.next_entities, self.range_arr)

        self.argument_logits, self.ret_hidden_argu_rep = self.judge.classify_argument(self.input_argument)
        self.final_logits_test = self.judge.get_logits_argument(self.input_hidden_argu_rep)
        self.test_state_agent_1 = tf.stack(test_state_agent_1)
        self.test_state_agent_2 = tf.stack(test_state_agent_2)
        logger.info('TF Graph creation done..')
        self.model_saver = tf.train.Saver(max_to_keep=2)

        if not restore:
            return tf.global_variables_initializer()
        else:
            return self.model_saver.restore(sess, restore)


    def initialize_pretrained_embeddings(self, sess):
        '''
        Initializes embeddings of the judge and agents with pretrained embeddings from other method.

        :param sess: Tf.Session. Session used to load the values.
        :return: None.
        '''
        list_special_relations = ['PAD','DUMMY_START_RELATION','NO_OP','UNK','END_OF_ARGUMENT']
        list_special_entities = ['PAD','UNK','QUERY_SUBJECT']

        if self.pretrained_embeddings_action != '0':
            logger.info("Using pretrained relation embeddings".upper())
            pre_trained_embeddings = pickle.load(open(self.pretrained_embeddings_action,'rb'))
            pretrained_dict = pickle.load(open(self.pretrained_rel_dict,'rb')) ## TODO: Change
            embeddings = np.zeros(self.judge.relation_lookup_table.shape)
            for relation, idx in self.relation_vocab.items():
                if relation in list_special_relations:
                    continue #This relations have no correspondance for embedding methods.
                embeddings[idx] = pre_trained_embeddings[pretrained_dict[relation]]

            ## Judge
            init_embeddings = sess.run(self.judge.relation_lookup_table)
            for relation in list_special_relations:
                embeddings[self.relation_vocab[relation]] = init_embeddings[self.relation_vocab[relation]]
            _ = sess.run((self.judge.relation_embedding_init),
                         feed_dict={self.judge.relation_embedding_placeholder: embeddings})

            ## Agent 1
            init_embeddings = sess.run(self.agent.relation_lookup_table_agent_1)
            for relation in list_special_relations:
                embeddings[self.relation_vocab[relation]] = init_embeddings[self.relation_vocab[relation]]
            _ = sess.run((self.agent.relation_embedding_init_agent_1),
                         feed_dict={self.agent.relation_embedding_placeholder_agent_1: embeddings})

            ## Agent 2
            init_embeddings = sess.run(self.agent.relation_lookup_table_agent_2)
            for relation in list_special_relations:
                embeddings[self.relation_vocab[relation]] = init_embeddings[self.relation_vocab[relation]]
            _ = sess.run((self.agent.relation_embedding_init_agent_2),
                         feed_dict={self.agent.relation_embedding_placeholder_agent_2: embeddings})
            logger.info("Pretrained relations successfully loaded")


        if self.pretrained_embeddings_entity != '0':
            logger.info("Using pretrained entities embeddings".upper())
            pre_trained_embeddings = pickle.load(open(self.pretrained_embeddings_entity,'rb'))
            pretrained_dict = pickle.load(open(self.pretrained_ent_dict,'rb')) ## TODO: Change
            embeddings = np.zeros(self.judge.entity_lookup_table.shape)
            for entity, idx in self.entity_vocab.items():
                if entity in list_special_entities :
                    continue #This entities have no correspondance for embedding methods.
                embeddings[idx] = pre_trained_embeddings[pretrained_dict[entity]]

            ## Judge
            init_embeddings = sess.run(self.judge.entity_lookup_table)
            for entity in list_special_entities:
                embeddings[self.entity_vocab[entity]] = init_embeddings[self.entity_vocab[entity]]
            _ = sess.run((self.judge.entity_embedding_init),
                         feed_dict={self.judge.entity_embedding_placeholder: embeddings})

            ## Agent 1
            init_embeddings = sess.run(self.agent.entity_lookup_table_agent_1)
            for entity in list_special_entities:
                embeddings[self.entity_vocab[entity]] = init_embeddings[self.entity_vocab[entity]]
            _ = sess.run((self.agent.entity_embedding_init_agent_1),
                         feed_dict={self.agent.entity_embedding_placeholder_agent_1: embeddings})

            ## Agent 2
            init_embeddings = sess.run(self.agent.entity_lookup_table_agent_2)
            for entity in list_special_entities:
                embeddings[self.entity_vocab[entity]] = init_embeddings[self.entity_vocab[entity]]
            _ = sess.run((self.agent.entity_embedding_init_agent_2),
                         feed_dict={self.agent.entity_embedding_placeholder_agent_2: embeddings})
            logger.info("Pretrained entities successfully loaded")


    def bp_judge(self, loss):
        '''
        Adds the backpropagation operation for the judge to the graph.

        :param loss: Tensor, []. Judge's loss for the episode.
        :return: Tf.Operation. Operation for the judge's backpropagation.
        '''

        tvars = tf.trainable_variables("judge")
        grads = tf.gradients(loss, tvars)
        grads, _ = tf.clip_by_global_norm(grads, self.grad_clip_norm)
        train_op = self.optimizer_judge.apply_gradients(zip(grads, tvars))
        with tf.control_dependencies([train_op]):   # see https://github.com/tensorflow/tensorflow/issues/1899
            self.dummy_judge = tf.constant(0)

        return train_op


    def bp(self, loss_1, loss_2):
        '''
        Adds the backpropagation operation for the agents to the graph.

        :param loss_1: Tensor, []. Pro agent's loss for the episode.
        :param loss_2: Tensor, []. Con agent's loss for the episode.
        :return: Tuple of two Tf.Operation. Opeartions for the agents' backpropagation.
        '''

        if not self.custom_baseline:
            self.baseline_1.update(tf.reduce_mean(self.cum_discounted_reward_1))
            self.baseline_2.update(tf.reduce_mean(self.cum_discounted_reward_2))

        tvars_1 = tf.trainable_variables("MLP_for_policy_1")
        tvars_2 = tf.trainable_variables("MLP_for_policy_2")


        grads_1 = tf.gradients(loss_1, tvars_1)
        grads_2 = tf.gradients(loss_2, tvars_2)

        grads_1, _ = tf.clip_by_global_norm(grads_1, self.grad_clip_norm)
        grads_2, _ = tf.clip_by_global_norm(grads_2, self.grad_clip_norm)

        train_op_1 = self.optimizer_agents.apply_gradients(zip(grads_1, tvars_1))
        train_op_2 = self.optimizer_agents.apply_gradients(zip(grads_2, tvars_2))

        with tf.control_dependencies(
                [train_op_1, train_op_2]):  # see https://github.com/tensorflow/tensorflow/issues/1899
            self.dummy = tf.constant(0,name='dummy_train_agents')
        return train_op_1, train_op_2


    def calc_cum_discounted_reward(self, rewards_1, rewards_2, which_agent_sequence):
        '''
        Calculates the discounted rewards of the agents.


        :param rewards_1: Numpy array of shape [Batch_size, path_length * num_arguments]. Array with the rewards of pro agent
        for every action it took.
        :param rewards_2: Numpy array of shape [Batch_size, path_length * num_arguments]. Array with hte rewards of con agent
        for every action it took.
        :param which_agent_sequence: Numpy array of shape [number_steps]. Information about which timestep corresponds to
        an action taken by either the pro agent (0) or the con agent (1).
        :return: Tuple of two numpy arrays, each [Batch_size, path_length * num_arguments]. Arrays with the discountinued
        rewards for both agents.
        '''

        r_1_index = -1
        r_2_index = -1
        running_add_1 = np.zeros([rewards_1.shape[0]])  # [B]
        no_paths_1 = np.count_nonzero(which_agent_sequence == 0.0)
        cum_disc_reward_1 = np.zeros([rewards_1.shape[0], no_paths_1])  # [B, T_1]

        running_add_2 = np.zeros([rewards_2.shape[0]])  # [B]
        no_paths_2 = np.count_nonzero(which_agent_sequence == 1.0)
        cum_disc_reward_2 = np.zeros([rewards_2.shape[0], no_paths_2])  # [B, T_2]

        prev_t = None
        for t in reversed(which_agent_sequence):
            if t == 0.0:
                if prev_t == 0:
                    running_add_1 = self.gamma * running_add_1 + rewards_1[:, r_1_index]
                else:
                    running_add_1 = rewards_1[:, r_1_index]
                cum_disc_reward_1[:, r_1_index] = running_add_1
                r_1_index -= 1
            if t == 1.0:
                if prev_t == 1:
                    running_add_2 = self.gamma * running_add_2 + rewards_2[:, r_2_index]
                else:
                    running_add_2 = rewards_2[:, r_2_index]
                cum_disc_reward_2[:, r_2_index] = running_add_2
                r_2_index -= 1
            prev_t = t

        return cum_disc_reward_1, cum_disc_reward_2

    def get_partial_run_setup(self, is_train_judge):
        '''
        Gets the elements needed to configure the partial run during training.

        Depending on the value of the is_train_judge_flag, the fetches will include either the dummy_operation for the
        agents' or the judge's backpropagation. Even if they are not called during the partial_run, giving the dummy_op
        to the configuration will automatically call the backpropagation and ruin the learning.

        The return values are used to configure the partial_run. They include:
            - fetches: List of tensors and operations. The values that will be fetched/queries during the partial run.
            - feeds: List of tensors and operations. The values that will be fed during the partial run.
            - feed_dict: List of dicts. Dictionaries for every timestep.

        :param is_train_judge: bool. Flag to tell if the judge or the agents should be trained.
        :return: Triple of three elements.
        '''

        fetches = self.temp_logits_judge + self.per_example_loss + self.action_idx + [self.loss_1] + [self.loss_2] \
                  + self.per_example_logits + self.rewards_before_baseline \
                  + [self.loss_judge, self.final_logits_judge] + self.rewards_agents

        feeds = self.candidate_relation_sequence + self.candidate_entity_sequence + self.input_path + \
                [self.query_subject] + [self.query_relation] + [self.query_object] + [self.cum_discounted_reward_1] + [
                    self.cum_discounted_reward_2] + [self.range_arr] + self.entity_sequence + [
                    self.which_agent] + [self.labels] + self.which_agent_sequence + [self.learning_rate_judge_ph] + \
                [self.random_flag]

        feed_dict = [{} for _ in range(self.number_steps)]
        feed_dict[0][self.labels] = None
        feed_dict[0][self.query_subject] = None
        feed_dict[0][self.query_relation] = None
        feed_dict[0][self.query_object] = None
        feed_dict[0][self.learning_rate_judge_ph] = None

        feed_dict[0][self.range_arr] = np.arange(self.batch_size)
        for i in range(self.number_steps):
            feed_dict[i][self.input_path[i]] = np.zeros(self.batch_size)  # placebo
            feed_dict[i][self.candidate_relation_sequence[i]] = None
            feed_dict[i][self.candidate_entity_sequence[i]] = None
            feed_dict[i][self.entity_sequence[i]] = None
            feed_dict[i][self.which_agent_sequence[i]] = None

        if is_train_judge:
            fetches.append(self.dummy_judge)
        else:
            fetches.append(self.dummy)

        return fetches, feeds, feed_dict


    def train(self, sess):
        '''
        Trains R2D2.

        :param sess: Tf.Session(). Session where the graph values have been initialzied.
        :return: None
        '''

        self.batch_counter = 0

        debate_printer = Debate_Printer(self.output_dir, self.train_environment.grapher, self.num_rollouts)
        for episode in self.train_environment.get_episodes():
            is_train_judge = (self.batch_counter // self.train_judge_every) % 2 == 0 \
                             or self.batch_counter >= self.rounds_sub_training
            fetches, feeds, feed_dict = self.get_partial_run_setup(is_train_judge)
            which_agent_list = []
            logger.info("BATCH COUNTER: {} ".format(self.batch_counter))
            self.batch_counter += 1

            feed_dict[0][self.query_subject] = episode.get_query_subjects()  # check getters
            feed_dict[0][self.query_relation] = episode.get_query_relation()
            feed_dict[0][self.query_object] = episode.get_query_objects()
            feed_dict[0][self.learning_rate_judge_ph] = self.learning_rate_judge  # check getters
            feed_dict[0][self.random_flag] = self.batch_counter < self.train_judge_every

            episode_answers = episode.get_labels()

            debate_printer.create_debates(episode.get_query_subjects(), episode.get_query_relation(), episode.get_query_objects(),
                                          episode.get_labels())


            feed_dict[0][self.labels] = episode_answers

            loss_before_regularization = []
            logits = []
            i = 0

            h = sess.partial_run_setup(fetches=fetches, feeds=feeds)

            debate_printer_rel_list = []
            debate_printer_ent_list = []
            for arguments in range(self.number_arguments):
                state = episode.reset_initial_state()
                for path_num in range(self.path_length):
                    which_agent_list.append(0.0)
                    feed_dict[i][self.which_agent_sequence[i]] = 0.0
                    feed_dict[i][self.candidate_relation_sequence[i]] = state['next_relations']
                    feed_dict[i][self.candidate_entity_sequence[i]] = state['next_entities']
                    feed_dict[i][self.entity_sequence[i]] = state['current_entities']
                    temp_logits_judge, per_example_loss, per_example_logits, idx, rewards, print_rewards = sess.partial_run(h, [
                        self.temp_logits_judge[i],
                        self.per_example_loss[i],
                        self.per_example_logits[i],
                        self.action_idx[i],
                        self.rewards_agents[i],
                        self.rewards_before_baseline[i]],
                                                        feed_dict=feed_dict[i])
                    loss_before_regularization.append(per_example_loss)
                    rel_string, ent_string = debate_printer.get_action_rel_ent(idx, state)
                    debate_printer_rel_list.append(rel_string)
                    debate_printer_ent_list.append(ent_string)
                    state = episode(idx)
                    i += 1
                    logits.append((0, rewards))


                debate_printer.create_arguments(debate_printer_rel_list, debate_printer_ent_list, rewards, True)
                debate_printer_rel_list.clear()
                debate_printer_ent_list.clear()

                state = episode.reset_initial_state()

                for path_num in range(self.path_length):
                    which_agent_list.append(1.0)
                    feed_dict[i][self.which_agent_sequence[i]] = 1.0
                    feed_dict[i][self.candidate_relation_sequence[i]] = state['next_relations']
                    feed_dict[i][self.candidate_entity_sequence[i]] = state['next_entities']
                    feed_dict[i][self.entity_sequence[i]] = state['current_entities']
                    temp_logits_judge, per_example_loss, per_example_logits, idx, rewards, print_rewards = sess.partial_run(h, [
                        self.temp_logits_judge[i],
                        self.per_example_loss[i],
                        self.per_example_logits[i],
                        self.action_idx[i],
                        self.rewards_agents[i],
                        self.rewards_before_baseline[i]],
                                                                                                    feed_dict=feed_dict[
                                                                                                        i])
                    loss_before_regularization.append(per_example_loss)
                    rel_string, ent_string = debate_printer.get_action_rel_ent(idx, state)
                    debate_printer_rel_list.append(rel_string)
                    debate_printer_ent_list.append(ent_string)
                    state = episode(idx)
                    i += 1
                    logits.append((1,rewards))


                debate_printer.create_arguments(debate_printer_rel_list, debate_printer_ent_list, rewards, False)
                debate_printer_rel_list.clear()
                debate_printer_ent_list.clear()


            logits_judge = sess.partial_run(h, self.final_logits_judge)
            debate_printer.set_debates_final_logit(logits_judge)
            if is_train_judge:
                loss_judge, _ = sess.partial_run(h, [
                                                                      self.loss_judge,
                                                                      self.dummy_judge])

                self.learning_rate_judge = self.learning_rate_judge_init
            else:
                print("judge is NOT trained \n")

            predictions = logits_judge > 0
            if self.batch_counter % self.save_debate_every == 0:
                debate_printer.write('argument_train_{}.txt'.format(self.batch_counter))

            acc = np.mean(predictions == episode_answers)
            logger.info("Mean label === {}".format(np.mean(episode_answers)))
            logger.info("Acc === {}".format(acc))


            rewards_1, rewards_2 = episode.get_rewards(logits)
            logger.info("MEDIAN REWARD A1 === {}".format(np.mean(rewards_1)))
            logger.info("MEDIAN REWARD A2 === {}".format(np.mean(rewards_2)))

            # computed cumulative discounted reward
            cum_discounted_reward_1, cum_discounted_reward_2 = self.calc_cum_discounted_reward(rewards_1, rewards_2,
                                                                                               np.array(
                                                                                                   which_agent_list))  # [B, T]

            if not is_train_judge:
                _ = sess.partial_run(h,[self.dummy],
                                      feed_dict={self.cum_discounted_reward_1: cum_discounted_reward_1,
                                                 self.cum_discounted_reward_2: cum_discounted_reward_2})  # self.dummy seems to be a hack for training_op

            if self.batch_counter == self.rounds_sub_training:
                self.model_saver.save(sess, self.model_dir + '/unbiased_model/unbiased_model.ckpt')

            if self.batch_counter % self.eval_every == 0:
                self.test(sess, True)

            if os.name == 'posix':
               logger.info('Memory usage : %s (kb)' % resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)

            gc.collect()
            if self.batch_counter >= self.total_iterations:
                break

    def test(self, sess, is_dev_environment, save_model=False, best_threshold=None):
        '''
        Tests R2D2 with the value of the current weights.

        :param sess: Tf.Session. Session where the graph values have been traiend.
        :param is_dev_environment: bool. Flag to differentiate between using the dev set for validation and the test set
        for testing.
        :param save_model: bool. Flag to check if the model should be save regardless of performance.
        :param best_threshold: float. Threshold that maximizes the accuracy in the dev set. Used to compute the correct
        accuracy during testing. If None, the accuracy is maximized.
        :return: Float. Threshold that maximizes the accuracy if best_threshold was none. Otherwise, returns best_threshold.
        '''

        batch_counter = 0
        total_examples = 0
        mean_probs_list = []
        correct_answer_list = []
        sk_mean_logit_list = []
        sk_correct_answer_list = []
        logger.info("Test start")
        environment = self.dev_test_environment if is_dev_environment else self.test_test_environment
        hitsAt20 = 0
        hitsAt10 = 0
        hitsAt3 = 0
        hitsAt1 = 0
        mean_reciprocal_rank = 0
        mean_rank = 0
        debate_printer = Debate_Printer(self.output_dir, self.train_environment.grapher, self.test_rollouts, is_append=True)
        for episode in tqdm(environment.get_episodes()):
            logger.info("TEST COUNTER: {} ".format(batch_counter))
            feed_dict = {}
            rep_argu_list = []

            batch_counter += 1

            temp_batch_size = episode.no_examples

            self.qr = episode.get_query_relation()
            feed_dict[self.query_relation] = self.qr
            # get initial state

            agent_mem_1, agent_mem_2 = self.agent.get_init_state_array(temp_batch_size)

            previous_relation = np.ones((temp_batch_size * self.test_rollouts,), dtype='int64') * self.relation_vocab[
                'DUMMY_START_RELATION']

            episode_answers = episode.get_labels()


            feed_dict[self.range_arr] = np.arange(temp_batch_size * self.test_rollouts)
            feed_dict[self.input_path[0]] = np.zeros(temp_batch_size * self.test_rollouts)
            feed_dict[self.labels] = episode_answers
            feed_dict[self.random_flag] = False

            feed_dict[self.query_subject] = episode.get_query_subjects()  # check getters
            feed_dict[self.query_relation] = episode.get_query_relation()
            feed_dict[self.query_object] = episode.get_query_objects()

            i = 0
            self.log_probs = np.zeros((temp_batch_size * self.test_rollouts,)) * 1.0

            debate_printer.create_debates(episode.get_query_subjects(), episode.get_query_relation(), episode.get_query_objects(),
                                          episode.get_labels())

            input_argument = 0 #Dummy initial value
            debate_printer_rel_list = []
            debate_printer_ent_list = []
            for argument_num in range(self.number_arguments):
                state = episode.reset_initial_state()
                # for each time step
                for path_num in range(self.path_length):
                    feed_dict[self.which_agent] = 0.0
                    feed_dict[self.next_relations] = state['next_relations']
                    feed_dict[self.next_entities] = state['next_entities']
                    feed_dict[self.current_entities] = state['current_entities']
                    feed_dict[self.prev_relation] = previous_relation
                    feed_dict[self.t] = i
                    feed_dict[self.input_argument] = input_argument
                    feed_dict[self.prev_state_agent_1] = agent_mem_1
                    feed_dict[self.prev_state_agent_2] = agent_mem_2
                    loss, agent_mem_1, \
                        agent_mem_2, test_scores, test_action_idx, chosen_relation, input_argument = sess.run(
                            [self.test_loss,
                                self.test_state_agent_1,
                                self.test_state_agent_2, self.test_logits,
                                self.test_action_idx, self.chosen_relation, self.ret_argument],
                            feed_dict=feed_dict)

                    previous_relation = chosen_relation
                    rel_string, ent_string = debate_printer.get_action_rel_ent(test_action_idx, state)
                    debate_printer_rel_list.append(rel_string)
                    debate_printer_ent_list.append(ent_string)
                    state = episode(test_action_idx)
                    i += 1

                #Append the logits the last argument produced.
                feed_dict[self.input_argument] = input_argument
                logits_last_argument, hidden_rep_argu = sess.run([self.argument_logits, self.ret_hidden_argu_rep],feed_dict=feed_dict)
                rewards = logits_last_argument
                rep_argu_list.append(hidden_rep_argu)
                debate_printer.create_arguments(debate_printer_rel_list, debate_printer_ent_list, rewards, True)

                debate_printer_rel_list.clear()
                debate_printer_ent_list.clear()

                state = episode.reset_initial_state()
                for path_num in range(self.path_length):
                    feed_dict[self.which_agent] = 1.0
                    feed_dict[self.next_relations] = state['next_relations']
                    feed_dict[self.next_entities] = state['next_entities']
                    feed_dict[self.current_entities] = state['current_entities']
                    feed_dict[self.prev_relation] = previous_relation
                    feed_dict[self.t] = i
                    feed_dict[self.input_argument] = input_argument
                    feed_dict[self.prev_state_agent_1] = agent_mem_1
                    feed_dict[self.prev_state_agent_2] = agent_mem_2
                    loss, agent_mem_1, \
                    agent_mem_2, test_scores, test_action_idx, chosen_relation, input_argument = sess.run(
                        [self.test_loss,
                         self.test_state_agent_1,
                         self.test_state_agent_2, self.test_logits,
                         self.test_action_idx, self.chosen_relation, self.ret_argument],
                        feed_dict=feed_dict)
                    previous_relation = chosen_relation
                    rel_string, ent_string = debate_printer.get_action_rel_ent(test_action_idx, state)
                    debate_printer_rel_list.append(rel_string)
                    debate_printer_ent_list.append(ent_string)
                    state = episode(test_action_idx)
                    i += 1

                feed_dict[self.input_argument] = input_argument
                logits_last_argument, hidden_rep_argu = sess.run([self.argument_logits, self.ret_hidden_argu_rep], feed_dict=feed_dict)
                rewards = logits_last_argument
                rep_argu_list.append(hidden_rep_argu)
                debate_printer.create_arguments(debate_printer_rel_list, debate_printer_ent_list, rewards, False)

                debate_printer_rel_list.clear()
                debate_printer_ent_list.clear()

            mean_argu_rep = np.mean(np.concatenate([np.expand_dims(rep_argu,axis=-1) for rep_argu in rep_argu_list],axis=-1),axis=-1)
            logits_judge = sess.run(self.final_logits_test,feed_dict={self.input_hidden_argu_rep: mean_argu_rep})

            reshaped_logits_judge = logits_judge.reshape((temp_batch_size,self.test_rollouts)) #[temp_batch_size, rollouts]
            reshaped_answer = episode_answers.reshape((temp_batch_size, self.test_rollouts))
            correct_answer_list.append(reshaped_answer[:,[0]])
            probs_judge = sig(reshaped_logits_judge)
            mean_probs = np.mean(probs_judge,axis=1,keepdims=True) #[temp_batch_size,1]
            mean_probs_list.append(mean_probs)
            reshaped_mean_probs = mean_probs.reshape((-1,temp_batch_size)) #[1,num false facts]
            idx_final_logits = np.argsort(reshaped_mean_probs,axis=1)

            mean_logits = np.mean(reshaped_logits_judge,axis=1)
            sk_mean_logit_list.append(mean_logits)
            sk_correct_answer_list.append(reshaped_answer[:, 0])

            debate_printer.create_best_debates()
            debate_printer.set_debates_final_logit(mean_logits)
            debate_printer.write('argument_test_new.txt')

            for fact in idx_final_logits:
                ans_rank = None
                rank = 0
                for ix in np.flip(fact,axis=0):
                    if ix == 0:
                        ans_rank = rank
                        break
                    rank += 1
                mean_reciprocal_rank += 1/(ans_rank+1) if ans_rank != None else 0
                mean_rank += ans_rank + 1 if ans_rank != None else reshaped_mean_probs.shape[1] + 1 #If, for whatever reason, the right entity was not ranked, we consider that as the last rank + 1
                if rank < 20:
                    hitsAt20 += 1
                    if rank < 10:
                        hitsAt10 += 1
                        if rank < 3:
                            hitsAt3 += 1
                            if rank == 0:
                                hitsAt1 += 1

            total_examples += reshaped_mean_probs.shape[0]

        sk_correct_answer_list = np.concatenate(sk_correct_answer_list).astype(int)
        sk_mean_logit_list = np.concatenate(sk_mean_logit_list)
        precision, recall, thresholds = sklearn.metrics.precision_recall_curve(sk_correct_answer_list, sk_mean_logit_list)
        auc_pr = sklearn.metrics.auc(recall, precision)
        fpr, tpr, thresholds = sklearn.metrics.roc_curve(sk_correct_answer_list, sk_mean_logit_list)
        auc_roc = sklearn.metrics.auc(fpr, tpr)

        best_acc = -1
        # compute accuracy with the best threshold:
        if best_threshold is None:
            for threshold in thresholds:  # We use the threshold given by roc_auc_score
                binary_preds = np.greater(sk_mean_logit_list, threshold).astype(int)
                acc = sklearn.metrics.accuracy_score(binary_preds, sk_correct_answer_list)
                if best_acc < acc:
                    best_acc = acc
                    best_threshold = threshold
        else:
            for threshold in thresholds:  # We use the threshold given by roc_auc_score
                binary_preds = np.greater(sk_mean_logit_list, threshold).astype(int)
                acc = sklearn.metrics.accuracy_score(binary_preds, sk_correct_answer_list)
                if best_acc < acc:
                    best_acc = acc
                    wrong_best_threshold = threshold

            logger.info("NOT BEST ACC === {}".format(best_acc))
            logger.info("NOT BEST THRESHOLD === {}".format(wrong_best_threshold))
            binary_preds = np.greater(sk_mean_logit_list, best_threshold).astype(int)
            best_acc = sklearn.metrics.accuracy_score(binary_preds, sk_correct_answer_list)


        logger.info("========== SKLEARN METRICS =============")
        logger.info("Best Threshold === {}".format(best_threshold))
        logger.info("Acc === {}".format(best_acc))
        logger.info("AUC_PR === {}".format(auc_pr))
        logger.info("AUC_ROC === {}".format(auc_roc))
        logger.info("========================================")

        if self.is_use_fixed_false_facts:
            if save_model or best_acc > self.best_metric:
                self.save_path = self.model_saver.save(sess, self.model_dir + "model" + '.ckpt')
            self.best_metric = best_acc if best_acc > self.best_metric else self.best_metric
            self.best_threshold = best_threshold
        else:
            if save_model or mean_reciprocal_rank > self.best_metric:
                self.save_path = self.model_saver.save(sess, self.model_dir + "model" + '.ckpt')
            self.best_metric = mean_reciprocal_rank if mean_reciprocal_rank > self.best_metric else self.best_metric

        logger.info("Hits@20 === {}".format(hitsAt20 / total_examples))
        logger.info("Hits@10 === {}".format(hitsAt10 / total_examples))
        logger.info("Hits@3 === {}".format(hitsAt3 / total_examples))
        logger.info("Hits@1 === {}".format(hitsAt1 / total_examples))
        logger.info("MRR === {}".format(mean_reciprocal_rank / total_examples))
        logger.info("MR === {}".format(mean_rank / total_examples))

        return best_threshold

def main():
    '''
    Runs an experiment or evaluates a pretrained model based on the value of the load_model option.

    It is assumed that there is only one value for each hyperparameter load_model is 1 (testing).
    For training, if at least one hyperparameter has more than one value, a grid search over the whole range is performed.
    In that case, trained models are only saved if they outperform the best performing model in the development set. Finally,
    the best performing model at the end of the sweep is tested on the test set.

    :return: None.
    '''

    option = read_options()
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter('%(asctime)s: [ %(message)s ]',
                            '%m/%d/%Y %I:%M:%S %p')
    console = logging.StreamHandler()
    console.setFormatter(fmt)
    logfile = None
    logger.addHandler(console)
    # read the vocab files, it will be used by many classes hence global scope
    logger.info('reading vocab files...')
    relation_vocab = json.load(open(option['vocab_dir'] + '/relation_vocab.json'))
    entity_vocab = json.load(open(option['vocab_dir'] + '/entity_vocab.json'))
    mid_to_name = json.load(open(option['vocab_dir'] + '/fb15k_names.json')) \
        if os.path.isfile(option['vocab_dir'] + '/fb15k_names.json') else None
    logger.info('Reading mid to name map')
    logger.info('Done..')
    logger.info('Total number of entities {}'.format(len(entity_vocab)))
    logger.info('Total number of relations {}'.format(len(relation_vocab)))
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = False
    config.log_device_placement = False

    if not option['load_model']:

        for key, val in option.items():
            if not isinstance(val, list):
                option[key] = [val]

        for permutation in ParameterGrid(option):
            best_permutation = None
            best_metric = 0

            current_time = datetime.datetime.now()
            current_time = current_time.strftime('%y_%b_%d__%H_%M_%S')
            permutation['output_dir'] = permutation['base_output_dir'] + '/' + str(current_time) + '__' + str(uuid.uuid4())[
                                                                                                      :4] + '_' + str(
                permutation['path_length']) + '_' + str(permutation['beta']) + '_' + str(
                permutation['test_rollouts']) + '_' + str(
                permutation['Lambda'])

            permutation['model_dir'] = permutation['output_dir'] + '/' + 'model/'

            permutation['load_model'] = (permutation['load_model'] == 1)

            ##Logger##
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
            logfile = logging.FileHandler(permutation['output_dir'] + '/log.txt', 'w')
            logfile.setFormatter(fmt)
            logger.addHandler(logfile)
            permutation['relation_vocab'] = relation_vocab
            permutation['entity_vocab'] = entity_vocab
            permutation['mid_to_name'] = mid_to_name

            # Training
            trainer = Trainer(permutation, best_metric)
            with tf.Session(config=config) as sess:
                sess.run(trainer.initialize())
                trainer.initialize_pretrained_embeddings(sess=sess)
                trainer.train(sess)

            if trainer.best_metric > best_metric or best_permutation == None:
                best_acc = trainer.best_metric
                best_threshold = trainer.best_threshold
                best_permutation = permutation
            tf.reset_default_graph()

        #Test best model
        current_time = datetime.datetime.now()
        current_time = current_time.strftime('%y_%b_%d__%H_%M_%S')
        best_permutation['output_dir'] = best_permutation['base_output_dir'] + '/' + str(current_time) + '__Test__' + str(uuid.uuid4())[
                                                                                                  :4] + '_' + str(
            best_permutation['path_length']) + '_' + str(best_permutation['beta']) + '_' + str(
            best_permutation['test_rollouts']) + '_' + str(
            best_permutation['Lambda'])

        best_permutation['old_model_dir'] = best_permutation['model_dir']
        best_permutation['model_dir'] = best_permutation['output_dir'] + '/' + 'model/'

        best_permutation['load_model'] = (best_permutation['load_model'] == 1)

        ##Logger##
        os.makedirs(best_permutation['output_dir'])
        os.mkdir(best_permutation['model_dir'])
        with open(best_permutation['output_dir'] + '/config.txt', 'w') as out:
            pprint(best_permutation, stream=out)

        # print and return
        maxLen = max([len(ii) for ii in best_permutation.keys()])
        fmtString = '\t%' + str(maxLen) + 's : %s'
        print('Arguments:')
        for keyPair in sorted(best_permutation.items()):
            if not keyPair[0].endswith('_vocab') and not keyPair[0] == 'mid_to_name' : print(fmtString % keyPair)
        logger.removeHandler(logfile)
        logfile = logging.FileHandler(best_permutation['output_dir'] + '/log.txt', 'w')
        logfile.setFormatter(fmt)
        logger.addHandler(logfile)
        trainer = Trainer(best_permutation, best_acc)

        with tf.Session(config=config) as sess:
            trainer.initialize(best_permutation['old_model_dir'] + "model" + '.ckpt',sess)
            trainer.test(sess,False,True, best_threshold)
    else:
        logger.info("Skipping training")
        logger.info("Loading model from {}".format(option["model_load_dir"]))

        for key, value in option.items():
            if isinstance(value, list):
                if len(value) == 1:
                    option[key] = value[0]
                else:
                    raise ValueError("Parameter {} has more than one value in the config file.".format(key))

        current_time = datetime.datetime.now()
        current_time = current_time.strftime('%y_%b_%d__%H_%M_%S')
        option['output_dir'] = option['base_output_dir'] + '/' + str(current_time) + '__Test__' + str(
            uuid.uuid4())[
                                                                                                    :4] + '_' + str(
            option['path_length']) + '_' + str(option['beta']) + '_' + str(
            option['test_rollouts']) + '_' + str(
            option['Lambda'])

        option['model_dir'] = option['output_dir'] + '/' + 'model/'
        os.makedirs(option['output_dir'])
        os.mkdir(option['model_dir'])
        with open(option['output_dir'] + '/config.txt', 'w') as out:
            pprint(option, stream=out)

        logger.removeHandler(logfile)
        logfile = logging.FileHandler(option['output_dir'] + '/log.txt', 'w')
        logfile.setFormatter(fmt)
        logger.addHandler(logfile)
        option['relation_vocab'] = relation_vocab
        option['entity_vocab'] = entity_vocab
        option['mid_to_name'] = mid_to_name
        trainer = Trainer(option, 0)

        with tf.Session(config=config) as sess:
            trainer.initialize(option['model_load_dir'] + "model" + '.ckpt',sess)
            if option['is_use_fixed_false_facts']:
                best_threshold = trainer.test(sess, True, False)
                trainer.test(sess, False, True, best_threshold=best_threshold)
            else:
                trainer.test(sess,False,True)


if __name__ == '__main__':
    main()
