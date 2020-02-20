from __future__ import absolute_import
from __future__ import division
import numpy as np
from code.data.feed_data import RelationEntityBatcher
from code.data.grapher import RelationEntityGrapher
import logging

logger = logging.getLogger()


class Episode(object):
    '''
    Class representing an episode for reinforcement learning. Adapted from https://github.com/shehzaadzd/MINERVA.

    An episode is a set of debates defined by the query triple, label of the triple, current entities of the agents and
    the knowledge graph that defines which actions are available for the agents at each debate step. This is encapsulated
    in self.state, a dictionary containing the next relations and entities available for the agent from current entities.
    '''

    def __init__(self, graph, data, params):
        '''
        Initializes an episode.

        :param graph: RelationEntityGrapher. Graph for the episode that defines which actions the agents might take.
        :param data: Tuple of 4 elements. Contains, in order, the query's subjects, relations, objects, labels and all
        correct answers for the subject-relation pairs.
        :param params: Tuple of 2. Number of rollouts and mode (either 'train', 'test' or 'dev'), respectively.
        '''
        self.grapher = graph
        num_rollouts, mode = params
        self.mode = mode
        self.num_rollouts = num_rollouts
        self.current_hop = 0
        start_entities, query_relation,  end_entities, labels, all_answers = data
        self.no_examples = start_entities.shape[0]
        start_entities = np.repeat(start_entities, self.num_rollouts)
        batch_query_relation = np.repeat(query_relation, self.num_rollouts)
        end_entities = np.repeat(end_entities, self.num_rollouts)
        self.start_entities = start_entities
        self.end_entities = end_entities
        self.current_entities = np.array(start_entities)
        self.query_relation = batch_query_relation
        self.all_answers = all_answers
        self.labels = np.repeat(labels, self.num_rollouts)

        next_actions = self.grapher.return_next_actions(self.current_entities, self.start_entities, self.query_relation,
                                                        self.end_entities, self.labels, self.all_answers,
                                                        self.num_rollouts)
        self.state = {}
        self.state['next_relations'] = next_actions[:, :, 1]
        self.state['next_entities'] = np.where(next_actions[:, :, 0] == np.repeat(np.expand_dims(self.start_entities,axis=-1),next_actions.shape[1],axis=-1),
                          self.grapher.get_placeholder_subject(), next_actions[:, :, 0])
        self.state['current_entities'] = np.where(self.current_entities == self.start_entities,self.grapher.get_placeholder_subject(),self.current_entities)

        self.init_state = dict(self.state)


    def reset_initial_state(self):
        '''
        Returns the initial state of the episode, i.e. simulates an agent going back to the starting entity.
        :return: Dict. Initial state of the episode.
        '''

        self.state = dict(self.init_state)
        return self.state

    def get_query_relation(self):
        '''
        Getter for the query's relations for the episode.
        :return: Numpy array, [batch_size]. Query's relations for the episode.
        '''

        return self.query_relation

    def get_default_query_subject(self):
        '''
        Getter for the placeholder string used to signal the query's subject.
        :return: String. Placeholder name for the query's subject.
        '''

        return self.grapher.QUERY_SUBJECT_NAME

    def get_query_subjects(self):
        '''
        Getter for the query's subjects of the episode.
        :return: Numpy array, [batch_size]. Query's subjects.
        '''

        return self.start_entities

    def get_query_objects(self):
        '''
        Getter for the query's objects of the episode.
        :return: Numpy array, [batch_size]. Query's objects.
        '''

        return self.end_entities

    def get_labels(self):
        '''
        Getter for the query's labels of the episode.

        Unlike the other getters, this returns a numpy array of different dimensionality. This is because the graph structure
        requires the labels shape to be different.

        :return: Numpy array, [batch_size,1]. Query's labels.
        '''

        return np.expand_dims(self.labels,1)

    def get_rewards(self, logits_sequence):
        '''
        Computes and gets the rewards of the agent for the episode.

        :param logits_sequence: List of tuples. The tuples have two elements. The first one is either 0 or 1 to signal
        to which agent the logit belongs. The second is the value of the logit as a [batch_size,1] numpy array.
        :return: Tuple of two numpy array, each [batch_size, path_length * num_arguments]. Arrays with the rewards for
        the agents for each action they took.
        '''

        rewards_1 = []
        rewards_2 = []
        for which_agent, logit in logits_sequence:
            if not which_agent:
                rewards_1.append(logit)
            else:
                rewards_2.append(logit)
        rewards_1 = np.stack(rewards_1, axis=1)
        rewards_2 = np.stack(rewards_2, axis=1)
        rewards_1 = np.squeeze(rewards_1, axis=-1)
        rewards_2 = -np.squeeze(rewards_2, axis=-1)

        return rewards_1, rewards_2


    def __call__(self, action):
        '''
        Call method that simulates a transition on the knowledge graph defined by action.

        :param action: Numpy array, [batch_size]. Each entry an action number as defined in the grapher file.
        :return: Dict. Updated state of the episode after taking actions.
        '''

        self.current_hop += 1
        true_next_entities = np.where(self.state['next_entities'] == self.grapher.get_placeholder_subject(),
                                      np.repeat(np.expand_dims(self.start_entities,axis=-1),self.state['next_entities'].shape[1],axis=-1),
                                      self.state['next_entities'])
        self.current_entities = true_next_entities[np.arange(self.no_examples*self.num_rollouts), action]

        next_actions = self.grapher.return_next_actions(self.current_entities, self.start_entities, self.query_relation,
                                                        self.end_entities, self.labels, self.all_answers,
                                                        self.num_rollouts)

        self.state['next_relations'] = next_actions[:, :, 1]
        self.state['next_entities'] = np.where(next_actions[:, :, 0] == np.repeat(np.expand_dims(self.start_entities,axis=-1),next_actions.shape[1],axis=-1),
                          self.grapher.get_placeholder_subject(), next_actions[:, :, 0])
        self.state['current_entities'] = np.where(self.current_entities == self.start_entities,self.grapher.get_placeholder_subject(),self.current_entities)
        return self.state


class env(object):
    '''
    Class representing an environment setting for the model.

    An environment contains the needed grapher and batchers and is capable of generating episodes.
    '''

    def __init__(self, params, mode='train'):
        '''
        Initializes an environment.

        :param params: Dict. Parameters of the experiment.
        :param mode: String. Either 'train', 'test' or 'dev'. Mode for the environment.
        '''

        self.num_rollouts = params['num_rollouts']
        self.mode = mode
        self.test_rollouts = params['test_rollouts']
        self.rounds_sub_training = params['rounds_sub_training']
        input_dir = params['data_input_dir']
        if mode == 'train':
            self.batcher = RelationEntityBatcher(input_dir=input_dir, batch_size=params['batch_size'],
                                                 entity_vocab=params['entity_vocab'],
                                                 relation_vocab=params['relation_vocab'],
                                                 is_use_fixed_false_facts=params['is_use_fixed_false_facts'],
                                                 num_false_facts=params['false_facts_train'],
                                                 rounds_sub_training=self.rounds_sub_training)
        else:
            self.batcher = RelationEntityBatcher(input_dir=input_dir, batch_size=params['batch_size'],
                                                 entity_vocab=params['entity_vocab'],
                                                 relation_vocab=params['relation_vocab'],
                                                 is_use_fixed_false_facts=params['is_use_fixed_false_facts'], mode=mode)

        self.grapher = RelationEntityGrapher(data_input_dir=params['data_input_dir'],
                                             mode = mode,
                                             max_num_actions=params['max_num_actions'],
                                             entity_vocab=params['entity_vocab'],
                                             relation_vocab=params['relation_vocab'],
                                             mid_to_name=params['mid_to_name'])



    def get_episodes(self):
        '''
        Creates and yields an episode in the environment.

        :return: Episode for the environment.
        '''
        if self.mode == 'train':
            params = self.num_rollouts, self.mode
            for data in self.batcher.yield_next_batch_train():
                yield Episode(self.grapher, data, params)
        else:
            params = self.test_rollouts, self.mode
            for data in self.batcher.yield_next_batch_test():
                if data == None:
                    return
                yield Episode(self.grapher, data, params)
