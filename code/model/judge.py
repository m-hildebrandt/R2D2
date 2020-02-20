import tensorflow as tf
import numpy as np

class Judge():
    '''
    Class representing the Judge in R2D2. Adapted from the agent class from https://github.com/shehzaadzd/MINERVA.

    It evaluates the arguments that the agents presents an assigns them a score
    that is used to train the agents. Furthermore, it assigns the final prediction score to the whole debate.
    '''

    def __init__(self, params):
        '''
        Initializes the judge.

        :param params: Dict. Parameters of the experiment.
        '''

        self.path_length = params['path_length']
        self.action_vocab_size = len(params['relation_vocab'])
        self.entity_vocab_size = len(params['entity_vocab'])
        self.embedding_size = params['embedding_size']
        self.hidden_size = params['hidden_size']
        if params['use_entity_embeddings']:
            self.entity_initializer = tf.contrib.layers.xavier_initializer()
        else:
            self.entity_initializer = tf.zeros_initializer()
        self.train_entities = params['train_entity_embeddings']
        self.train_relations = params['train_relation_embeddings']
        self.hidden_layers = params['layers_judge']
        self.batch_size = params['batch_size'] * (1 + params['false_facts_train']) * params['num_rollouts']
        self.use_entity_embeddings = params['use_entity_embeddings']


        self.define_embeddings()

    def define_embeddings(self):
        '''
        Creates and adds the embeddings for the KG's relations and entities to the graph, as well as assign operations
        needed when using pre-trained embeddings.

        :return: None
        '''

        with tf.variable_scope("judge"):
            self.relation_embedding_placeholder = tf.placeholder(tf.float32,
                                                                 [self.action_vocab_size, self.embedding_size])

            self.relation_lookup_table = tf.get_variable("relation_lookup_table",
                                                         shape=[self.action_vocab_size, self.embedding_size],
                                                         dtype=tf.float32,
                                                         initializer=tf.contrib.layers.xavier_initializer(),
                                                         trainable=self.train_relations)
            self.relation_embedding_init = self.relation_lookup_table.assign(self.relation_embedding_placeholder)

        with tf.variable_scope("judge"):
            self.entity_embedding_placeholder = tf.placeholder(tf.float32,
                                                               [self.entity_vocab_size, self.embedding_size])
            self.entity_lookup_table = tf.get_variable("entity_lookup_table",
                                                       shape=[self.entity_vocab_size, self.embedding_size],
                                                       dtype=tf.float32,
                                                       initializer=self.entity_initializer,
                                                       trainable=self.train_entities)
            self.entity_embedding_init = self.entity_lookup_table.assign(self.entity_embedding_placeholder)

    def set_labels_placeholder(self,labels):
        '''
        Setter for the label's placeholder.
        :param labels: Tf.Placeholder, [None,1]. Placeholder that will be fed the labels of the episode's query.
        :return: None.
        '''

        self.labels = labels


    def action_encoder_judge(self, next_relations, next_entities):
        '''
        Encodes a an action into its embedded representation.

        Depending on the value of use_entity_embeddings, this corresponds to either taking only the picked relation's
        embedding or concatenating it with the picked target entity's embedding.

        :param next_relations: Tensor, [Batch_size]. Tensor with the ids of the picked relations.
        :param next_entities: Tensor, [Batch_size]. Tensor with the ids of the picked target entities. Only used
        if use_entity_embeddings is True.
        :return: Tensor. Depending on whether use_entity_embeddings is set to True or False, the shape will be
        [Batch_size, embedding_size * 2] or [Batch_size, embedding_size], respectively. Embedded representation of the picked action.
        '''

        with tf.variable_scope("lookup_table_edge_encoder"):
            relation_embedding = tf.nn.embedding_lookup(self.relation_lookup_table, next_relations)
            entity_embedding = tf.nn.embedding_lookup(self.entity_lookup_table, next_entities)
            if self.use_entity_embeddings:
                action_embedding = tf.concat([relation_embedding, entity_embedding], axis=-1)
            else:
                action_embedding = relation_embedding

        return action_embedding


    def extend_argument(self, argument, t, action_idx, next_relations, next_entities, range_arr):
        '''
        Extends an argument by adding the embedded representation of an action as defined by next_relations and next_entities.

        If t % self.path_length == 0 , the argument is not extended by the new action, but rather the new action is defined
        as the whole argument. This corresponds to adding the new action to an 'empty' argument.

        :param argument: Tensor, [Batch_size, None]. Argument to be extended.
        :param t: Tensor, []. Step number for the episode.
        :param action_idx: Tensor, [Batch_size]. Number of the selected action. Between 0 and max_num_actions
        :param next_relations: Tensor, [Batch_size, max_num_actions]. Contains the ids of all possible next picked relations.
        :param next_entities: Tensor, [Batch_size, max_num_actions]. Contains the ids of all possible next picked entities.
        :param range_arr: Tensor, [Batch_size]. Arange tensor used to properly select the correct next action.
        :return: [Batch_size, None]. Extended argument. The size of the second dimension is a multiple of the embedding
        size.
        '''

        with tf.variable_scope("judge"):
            with tf.variable_scope("extend_argument"):
                chosen_relation = tf.gather_nd(next_relations, tf.transpose(tf.stack([range_arr, action_idx])))  # [B,]
                chosen_entities = tf.gather_nd(next_entities, tf.transpose(tf.stack([range_arr, action_idx])))
                action_embedding = self.action_encoder_judge(chosen_relation, chosen_entities)

                argument = tf.cond(tf.equal(t % self.path_length, 0), lambda: action_embedding, lambda: tf.concat([argument, action_embedding],axis=-1))

        return argument


    def classify_argument(self, argument):
        '''
        Classifies arguments by computing a hidden representation and assigning logits.

        :param argument: Tensor, [Batch_size, path_length * 2  * embedding_size + 2 * embedding_size] or
        [Batch_sie, path_length * embedding_size + 2 * embedding_size], depending on whether use_entity_embeddings is
        True or False, respectively.
        :return: Tuple of 2 Tensors with shapes [Batch_size, 1] and [Batch_size, hidden_size], respectively. The first one
        contains the logits for the arguments of the batch. The second one, the hidden representation of the arguments.
        '''

        argument = tf.concat([argument, self.query_relation_embedding, self.query_object_embedding], axis=-1)

        #Tensorflow needs to know the shape information for the dense layer. However, this is lost during the extension
        #step for the arguments. As a workaround, the argument is "reshaped" to the right dimensions so it can be fed
        #to the dense layer.
        if self.use_entity_embeddings:
            argument = tf.reshape(argument,
                                    shape=[-1, self.path_length * 2 * self.embedding_size + 2 * self.embedding_size])
        else:
            argument = tf.reshape(argument,
                                  shape=[-1, self.path_length * self.embedding_size + 2 * self.embedding_size])

        with tf.variable_scope("judge"):
            with tf.variable_scope("mlp",reuse=tf.AUTO_REUSE):
                hidden = argument
                for i in range(self.hidden_layers - 1):
                    hidden = tf.layers.dense(hidden,self.hidden_size, activation=tf.nn.relu, name="layer_{}".format(i))
                hidden = tf.layers.dense(hidden, self.hidden_size, name='layer_{}'.format(self.hidden_layers - 1))
        logits = self.get_logits_argument(hidden)

        return logits, hidden

    def get_logits_argument(self,argument):
        '''
        Assings logits to an argument.

        :param argument: Tensor, [Batch_size, path_length * 2  * embedding_size + 2 * embedding_size] or
        [Batch_sie, path_length * embedding_size + 2 * embedding_size], depending on whether use_entity_embeddings is
        True or False, respectively.
        :return: Tensor, [Batch_size,1]. The logits of the arguments.
        '''

        with tf.variable_scope("judge"):
            with tf.variable_scope("classifier", reuse=tf.AUTO_REUSE):
                logits = tf.layers.dense(argument, self.hidden_size, name='classifier_judge_0', activation=tf.nn.relu)
                logits = tf.layers.dense(logits,1, name='classifier_judge_1')
        return logits

    def final_loss(self, rep_argu_list):
        '''
        Computes the final loss and final logits of the debates using all arguments presented.

        This is done by averaging the hidden representation of all arguments presented in the debate and passing the
        average through the classifier MLP.

        :param rep_argu_list: List of Tensors, each of shape [Batch_size, hidden_size]. The arguments are the hidden
        representation of the arguments.
        :return: Tuple of 2 Tensors. The first one is the loss of the episode and has shape []. The second one are the
        final logits of the debate and has shape [Batch_size,1].
        '''

        with tf.variable_scope('judge'):
            average_argu = tf.reduce_mean(tf.concat([ tf.expand_dims(rep_argu, axis=-1) for rep_argu in rep_argu_list],axis=-1),axis=-1)
        final_logit = self.get_logits_argument(average_argu)
        final_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=self.labels,logits=final_logit))

        return final_loss, final_logit


    def set_query_embeddings(self, query_subject, query_relation, query_object):
        '''
        Sets the judge's query information to the placeholders used by trainer.

        :param query_subject: Tensor, [None]. Placeholder that will be fed the query's subjects.
        :param query_relation: Tensor, [None]. Placeholder that will be fed the query's relations.
        :param query_object: Tensor, [None]. Placeholder that will be fed the query's objects.
        :return: None
        '''


        self.query_subject_embedding = tf.nn.embedding_lookup(self.entity_lookup_table, query_subject)
        self.query_relation_embedding = tf.nn.embedding_lookup(self.relation_lookup_table, query_relation)
        self.query_object_embedding = tf.nn.embedding_lookup(self.entity_lookup_table, query_object)
