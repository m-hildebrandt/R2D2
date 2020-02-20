import numpy as np
import tensorflow as tf


class Agent(object):
    '''
    Class for the agents in R2D2. Adapted from Adapted from the agent class from https://github.com/shehzaadzd/MINERVA.

    A single instance of Agent contains both the pro and con agent.
    '''


    def __init__(self, params, judge):
        '''
        Initializes the agents.
        :param params: Dict. Parameters of the experiment.
        :param judge: Judge. Instance of Judge that the agents present arguments to.
        '''

        self.judge = judge
        self.action_vocab_size = len(params['relation_vocab'])
        self.entity_vocab_size = len(params['entity_vocab'])
        self.embedding_size = params['embedding_size']
        self.ePAD = tf.constant(params['entity_vocab']['PAD'], dtype=tf.int32)
        self.rPAD = tf.constant(params['relation_vocab']['PAD'], dtype=tf.int32)
        self.train_entities = params['train_entity_embeddings']
        self.train_relations = params['train_relation_embeddings']
        self.test_rollouts = params['test_rollouts']
        self.path_length = params['path_length']
        self.batch_size = params['batch_size'] * (1 + params['false_facts_train']) * params['num_rollouts']
        self.dummy_start_label = tf.constant(
            np.ones(self.batch_size, dtype='int64') * params['relation_vocab']['DUMMY_START_RELATION'])

        self.hidden_layers = params['layers_agent']
        self.custom_baseline = params['custom_baseline']
        self.use_entity_embeddings = params['use_entity_embeddings']
        if self.use_entity_embeddings:
            self.entity_initializer = tf.contrib.layers.xavier_initializer()
            self.m = 2
        else:
            self.m = 1
            self.entity_initializer = tf.zeros_initializer()

        self.define_embeddings()
        self.define_agents_policy()

    def define_embeddings(self):
        '''
        For both agents, creates and adds the embeddings for the KG's relations and entities to the graph, as well as
        assign operations needed when using pre-trained embeddings.

        :return: None
        '''

        with tf.variable_scope("MLP_for_policy_1"):

            self.relation_embedding_placeholder_agent_1 = tf.placeholder(tf.float32,
                                                                         [self.action_vocab_size, self.embedding_size])

            self.relation_lookup_table_agent_1 = tf.get_variable("relation_lookup_table_agent_1",
                                                         shape=[self.action_vocab_size, self.embedding_size],
                                                         dtype=tf.float32,
                                                         initializer=tf.contrib.layers.xavier_initializer(),
                                                         trainable=self.train_relations)
            self.relation_embedding_init_agent_1 = self.relation_lookup_table_agent_1.assign(self.relation_embedding_placeholder_agent_1)

            self.entity_embedding_placeholder_agent_1 = tf.placeholder(tf.float32,
                                                               [self.entity_vocab_size, self.embedding_size])
            self.entity_lookup_table_agent_1 = tf.get_variable("entity_lookup_table_agent_1",
                                                       shape=[self.entity_vocab_size, self.embedding_size],
                                                       dtype=tf.float32,
                                                       initializer=self.entity_initializer,
                                                       trainable=self.train_entities)
            self.entity_embedding_init_agent_1 = self.entity_lookup_table_agent_1.assign(self.entity_embedding_placeholder_agent_1)

        with tf.variable_scope("MLP_for_policy_2"):

            self.relation_embedding_placeholder_agent_2 = tf.placeholder(tf.float32,
                                                                         [self.action_vocab_size, self.embedding_size])

            self.relation_lookup_table_agent_2 = tf.get_variable("relation_lookup_table_agent_2",
                                                         shape=[self.action_vocab_size, self.embedding_size],
                                                         dtype=tf.float32,
                                                         initializer=tf.contrib.layers.xavier_initializer(),
                                                         trainable=self.train_relations)
            self.relation_embedding_init_agent_2 = self.relation_lookup_table_agent_2.assign(self.relation_embedding_placeholder_agent_2)

            self.entity_embedding_placeholder_agent_2 = tf.placeholder(tf.float32,
                                                               [self.entity_vocab_size, self.embedding_size])
            self.entity_lookup_table_agent_2 = tf.get_variable("entity_lookup_table_agent_2",
                                                       shape=[self.entity_vocab_size, self.embedding_size],
                                                       dtype=tf.float32,
                                                       initializer=self.entity_initializer,
                                                       trainable=self.train_entities)
            self.entity_embedding_init_agent_2 = self.entity_lookup_table_agent_2.assign(self.entity_embedding_placeholder_agent_2)

    def define_agents_policy(self):
        '''
        Defines the agents' policy.

        After returning, self.policy_agent_1 and self.policy_agent_2 are callables that take as the last selected action
        and the LSTM-state and returns a hidden representation used for action selection.
        :return: None
        '''

        cells = []
        for _ in range(self.hidden_layers):
            cells.append(
                tf.contrib.rnn.LSTMCell(self.m * self.embedding_size, use_peepholes=True, state_is_tuple=True))
        self.policy_agent_1 = tf.contrib.rnn.MultiRNNCell(cells, state_is_tuple=True)

        cells = []
        for _ in range(self.hidden_layers):
            cells.append(
                tf.contrib.rnn.LSTMCell(self.m * self.embedding_size, use_peepholes=True, state_is_tuple=True))
        self.policy_agent_2 = tf.contrib.rnn.MultiRNNCell(cells, state_is_tuple=True)


    def format_state(self,state):
        '''
        Formats the cell- and hidden-state of the LSTM such that it can be fed to the LSTM cells.

        This is necessary as the state is fed as a tensor but the MultiRNNCell requires the state to be a tuple.

        :param state: Tensor, [hidden_layers_agent, 2, Batch_size, embedding_size * m],
        A tensor containing the hidden and cell state of for the every RNN Cell of the agents.

        :return: List of list of two tensors. Each entry has the two tensors that define the state of the LSTM.
        '''

        formated_state = tf.unstack(state, self.hidden_layers)
        formated_state = [tf.unstack(s,2) for s in formated_state]

        return formated_state

    def get_mem_shape(self):
        '''
        Returns the shape of the agent's LSTMCell.

        :return: Tuple of 4 ints.
        '''

        return (self.hidden_layers, 2, None, self.m * self.embedding_size)

    def get_init_state_array(self, temp_batch_size):
        '''
        Returns a numpy array with the initial state of the agent's LSTMCell.

        In other words, returns an array of 0 whose dimensionality corresponds to LSTMCell's state.

        :param temp_batch_size: int. Size of the temporary batch.

        :return: Tuple of two numpy arrays. The first array represents the initial state of the pro agent's LSTMCell.
        The second, the initial state of the con agent's LSTMCell.
        '''

        mem_agent = self.get_mem_shape()
        agent_mem_1 = np.zeros(
            (mem_agent[0], mem_agent[1], temp_batch_size * self.test_rollouts, mem_agent[3])).astype('float32')
        agent_mem_2 = np.zeros(
            (mem_agent[0], mem_agent[1], temp_batch_size * self.test_rollouts, mem_agent[3])).astype('float32')
        return agent_mem_1, agent_mem_2


    def policy(self, input_action, which_agent):
        '''
        Input_action is passed through either the pro or the con agent's LSTMCell and the output is returned.

        The output can be used for action selection. Which agent's LSTMCell is used depends on the value of which_agent.
        The state of the corresponding agent is then updated.

        :param input_action: Tensor, [Batch_size, input_action_size + 3 *embedding_size]. Tensor with an embedding
        representing the action. This is created by concatenating the embeddings of the action itself (either only the
        relation or the relation concatenated with the target entity, depending on the value of use_entity_embeddings)
        and the query's triple.
        :param which_agent: Tensor, []. Tells which agent's policy to use. 0 for pro agent, 1 for con agent.
        :return: Tensor, [Batch_size, embedding_size * m]. Output of the LSTMCell that is used for action selection.
        '''

        def policy_1():
            with tf.variable_scope("MLP_for_policy_1", reuse=tf.AUTO_REUSE):
                return self.policy_agent_1(input_action, self.state_agent_1)

        def policy_2():
            with tf.variable_scope("MLP_for_policy_2", reuse=tf.AUTO_REUSE):
                return self.policy_agent_2(input_action, self.state_agent_2)

        output, new_state = tf.cond(tf.equal(which_agent, 0.0), policy_1, policy_2)
        new_state_stacked = tf.stack(new_state)
        state_agent_1_stacked = tf.stack(self.state_agent_1)
        # We only want to update the state of the agent corresponding to the action. We do this through this arithmetic op.
        state_agent_1_stacked = (1-which_agent)*new_state_stacked + which_agent* state_agent_1_stacked
        self.state_agent_1 = self.format_state(state_agent_1_stacked)

        state_agent_2_stacked = tf.stack(self.state_agent_2)
        # We only want to update the state of the agent corresponding to the action. We do this through this arithmetic op.
        state_agent_2_stacked = which_agent*new_state_stacked + (1-which_agent)* state_agent_2_stacked
        self.state_agent_2 = self.format_state(state_agent_2_stacked)

        return output


    def action_encoder_agent(self, next_relations, current_entities, which_agent):
        '''
        Encodes all actions an agent has available to pick next.

        Depending on the value of use_entity_embeddings, this corresponds to either taking only the next relations'
        embeddings or concatenating them with the current_entities' embeddings. Which embedding table is used depends on
        the value of which_agent.

        :param next_relations: Tensor, [Batch_size, max_num_actions]. Tensor with the ids of the next relations.
        :param current_entities: Tensor, [Batch_size, max_num_actions]. Tensor with the ids of the current entities.
        Only used if use_entity_embeddings is True.
        :param which_agent: Tensor, []. Tells which agent's embeddings to use. 0 for pro agent, 1 for con agent.
        :return: Tensor, either [Batch_size, max_num_actions, embedding_size * m]
        This tensor contains for each batch sample the embedded representation of all actions the agent has available to
        choose next.
        '''

        with tf.variable_scope("lookup_table_edge_encoder"):
            f1 = lambda: tf.nn.embedding_lookup(self.relation_lookup_table_agent_1, next_relations)
            f2 = lambda: tf.nn.embedding_lookup(self.relation_lookup_table_agent_2, next_relations)
            relation_embedding = tf.case([(tf.equal(which_agent,0.0), f1)], default = f2)

            f1 = lambda: tf.nn.embedding_lookup(self.entity_lookup_table_agent_1, current_entities)
            f2 = lambda: tf.nn.embedding_lookup(self.entity_lookup_table_agent_2, current_entities)
            entity_embedding = tf.case([(tf.equal(which_agent,0.0), f1)], default = f2)

            if self.use_entity_embeddings:
                action_embedding = tf.concat([relation_embedding, entity_embedding], axis=-1)
            else:
                action_embedding = relation_embedding

        return action_embedding


    def set_query_embeddings(self, query_subject, query_relation, query_object):
        '''
        Sets the agents' query information to the placeholders used by trainer.

        :param query_subject: Tensor, [None]. Placeholder that will be fed the query's subjects.
        :param query_relation: Tensor, [None]. Placeholder that will be fed the query's relations.
        :param query_object: Tensor, [None]. Placeholder that will be fed the query's objects.
        :return: None
        '''
        self.query_subject_embedding_agent_1 = tf.nn.embedding_lookup(self.entity_lookup_table_agent_1, query_subject)
        self.query_relation_embedding_agent_1 = tf.nn.embedding_lookup(self.relation_lookup_table_agent_1, query_relation)
        self.query_object_embedding_agent_1 = tf.nn.embedding_lookup(self.entity_lookup_table_agent_1, query_object)

        self.query_subject_embedding_agent_2 = tf.nn.embedding_lookup(self.entity_lookup_table_agent_2, query_subject)
        self.query_relation_embedding_agent_2 = tf.nn.embedding_lookup(self.relation_lookup_table_agent_2, query_relation)
        self.query_object_embedding_agent_2 = tf.nn.embedding_lookup(self.entity_lookup_table_agent_2, query_object)

    def step(self, next_relations, next_entities, prev_state_agent_1,
             prev_state_agent_2, prev_relation,current_entities, range_arr, which_agent, random_flag):
        '''
        Computes a step for an agent during the debate.

        This means, this method is responsible for picking the next action of an agent by considering the options available
        (through next_relation and next_entities) and the current state of the agent's LSTMCell. More concretely,
        the agent picks its next action by pasing the previously selected action through the LSTMCell, thus updating
        its inner state. Then, an inner product between the new hidden state and the embeddings of the actions is used
        to parametrize a distribution from which the next action is computed.

        Additionally, this method computes the loss of the picked action, used for REINFORCE.

        This method returns different values that are needed for backpropagation, printing of the arguments or the next
        episode step. The returned values are:
            - loss: Tensor, [Batch_size]. Tensor with the loss of the selected actions. Used for REINFORCE.
            - self.state_agent_1: [hidden_layers_agent, 2, batch_size, m * embedding_size]. Hidden and cell state of the
            pro agent's LSTMCell. Updated if step was called so this agent picked an action.
            - self.state_agent_2: [hidden_layers_agent, 2, batch_size, m * embedding_size]. Hidden and cell state of the
            con agent's LSTMCell. Updated if step was called so this agent picked an action.
            - tf.nn.log_softmax(scores): Tensor, [Batch_size, max_num_actions]. Log probabilities of the action distri-
            bution for the step. Used for entropy regularization.
            - action_idx: Tensor, [Batch_size]. Tensor with the action number picked. Between 0 and max_num_actions.
            - chosen_relation: Tensor, [Batch_size]. Tensor with the chosen relations' id.

        :param next_relations: Tensor, [Batch_size, max_num_actions]. Tensor with the ids of the next relations.
        :param next_entities: Tensor, [Batch_size, max_num_actions]. Tensor with the ids of the current entities.
        :param prev_state_agent_1: Tensor, [hidden_layers_agent, 2, batch_size, m * embedding_size]. Tensor representing
        the previous hidden and cell states of pro agent's LSTMCell.
        :param prev_state_agent_2: Tensor, [hidden_layers_agent, 2, batch_size, m * embedding_size]. Tensor representing
        the previous hidden and cell states of con agent's LSTMCell.
        :param prev_relation: Tensor, [Batch_size]. Tensor with the ids of the last selected actions.
        :param current_entities: Tensor, [Batch_size]. Tensor with the ids of the current entity.
        :param range_arr: Tensor, [Batch_size]. Arange tensor used to properly select the correct next action.
        :param which_agent: Tensor, []. Tensor signaling which agent's turn it is.
        :param random_flag: Tensor, []. Boolean tensor used as a flag to either pick random actions or use the agent's policy.
        During the first part of training, the judge is trained with random actions.
        :return: Tuple of 6 elements.
        '''

        self.state_agent_1 = prev_state_agent_1
        self.state_agent_2 = prev_state_agent_2
        is_agent_1 = tf.equal(which_agent, 0.0)


        # Get state vector
        f1 = lambda: tf.nn.embedding_lookup(self.entity_lookup_table_agent_1, current_entities)
        f2 = lambda: tf.nn.embedding_lookup(self.entity_lookup_table_agent_2, current_entities)
        prev_entity = tf.case([(tf.equal(which_agent, 0.0), f1)], default=f2)

        f1 = lambda: tf.nn.embedding_lookup(self.relation_lookup_table_agent_1, prev_relation)
        f2 = lambda: tf.nn.embedding_lookup(self.relation_lookup_table_agent_2, prev_relation)
        prev_relation = tf.case([(tf.equal(which_agent, 0.0), f1)], default=f2)

        if self.use_entity_embeddings:
            state = tf.concat([prev_relation, prev_entity], axis=-1)
        else:
            state = prev_relation

        def get_policy_state():
            query_subject_embedding = tf.cond(is_agent_1, lambda: self.query_subject_embedding_agent_1, lambda: self.query_subject_embedding_agent_2)
            query_relation_embedding = tf.cond(is_agent_1, lambda: self.query_relation_embedding_agent_1, lambda: self.query_relation_embedding_agent_2)
            query_object_embedding = tf.cond(is_agent_1, lambda: self.query_object_embedding_agent_1, lambda: self.query_object_embedding_agent_2)

            state_query_concat = tf.concat([state, query_subject_embedding, query_relation_embedding, query_object_embedding], axis=-1)

            return state_query_concat

        # MLP for policy#
        candidate_action_embeddings = self.action_encoder_agent(next_relations, next_entities, which_agent)
        output = self.policy(get_policy_state(), which_agent)
        output_expanded = tf.expand_dims(output, axis=1)  # [B, 1, 2D]
        prelim_scores = tf.reduce_sum(tf.multiply(candidate_action_embeddings, output_expanded), axis=2)

        # Masking PAD actions

        comparison_tensor = tf.ones_like(next_relations, dtype=tf.int32) * self.rPAD
        mask = tf.equal(next_relations, comparison_tensor)
        dummy_scores = tf.ones_like(prelim_scores) * -99999.0
        scores = tf.where(mask, dummy_scores, prelim_scores)
        uni_scores = tf.where(mask, dummy_scores, tf.ones_like(prelim_scores))

        # 4 sample action
        action = tf.to_int32(tf.multinomial(logits=scores, num_samples=1))
        action = tf.cond(random_flag, lambda: tf.to_int32(tf.multinomial(logits=uni_scores, num_samples=1)), lambda: action)
        #action = tf.zeros_like(action)

        label_action = tf.squeeze(action, axis=1)
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=scores, labels=label_action)

        action_idx = tf.squeeze(action)
        chosen_relation = tf.gather_nd(next_relations, tf.transpose(tf.stack([range_arr, action_idx])))

        return loss, self.state_agent_1, self.state_agent_2,  tf.nn.log_softmax(scores), action_idx, chosen_relation



    def __call__(self, which_agent, candidate_relation_sequence, candidate_entity_sequence, current_entities,
                range_arr, T=3, random_flag=None):
        '''
        Call method that constructs a whole debate and returns relevant values for debate visualization.

        The return values are used for backpropagation, normalization and printing the arguments. They include:
            - loss_judge: Tensor, []. Final loss for the judge for the whole batch of debates.
            - final_logit_judge: Tensor, [Batch_size, 1]. Tensor with the logit the judge assigned to the debate.
            - all_temp_logits_judge: List of tensors, each [Batch_size, 1]. Each tensor contains the logit that the judge
            assigned to the argument. Because the judge only classifies complete arguments, when the argument is incomplete,
            a dummy value of 0 logits is assigned.
            - all_loss: List of tensors, each [Batch_size]. Each tensors represents the loss of the action the corresponding
            agent picked. Used for REINFORCE.
            - all_logits: List of tensors, each [Batch_size, max_num_actions]. Each tensor has the log probabilities
            of the action distribution for the corresponding step. Used for entropy regularization.
            - action_idx: List of tensors, each [Batch_size]. Each tensor has the action number picked for the correspon-
            ding timestep.
            - all_rewards_agents:  List of tensors, each [Batch_size, 1]. Each tensor ahs the rewards the agent got
            for the corresponding timestep. As the agents only get a reward based on complete arguments, before the
            argument is complete, the reward is set to 0.
            - all_rewards_before_baseline: List of tensors, each [Batch_size, 1]. The rewards the agent got, before
            substracting the baseline value.

        :param which_agent: List of tensors, each with shape []. Placeholders that will be fed information about which
        agent's turn it is.
        :param candidate_relation_sequence: List of tensors, each with shape [Batch_size, max_num_actions]. Placeholders
        that will be fed the next relations the agents can pick.
        :param candidate_entity_sequence: List of tensors, each with shape [Batch_size, max_num_actions]. Placeholders
        that will be fed the next entities the agents can pick.
        :param current_entities: List of tensors, each with shape [Batch_size]. Placeholders that will be fed the current
        entity the agent is at.
        :param range_arr: Tensor, [Batch_size]. Placeholder that will be fed an arange array to properly select
        the correct next action.
        :param T: int. The total number of steps in the debate, i.e. path_length * num_arguments * 2.
        :param random_flag: Tensor, []. Placeholder boolean tensor representing a flag to check if random actions should
        be picked.
        :return: Tuple of 8 elements.
        '''

        def get_prev_state_agents():
            prev_state_agent_1 = self.policy_agent_1.zero_state(batch_size=self.batch_size, dtype=tf.float32)
            prev_state_agent_2 = self.policy_agent_2.zero_state(batch_size=self.batch_size, dtype=tf.float32)
            return prev_state_agent_1, prev_state_agent_2

        prev_relation = self.dummy_start_label #TODO: Use dummy label to signal end of argument to agents?
        argument = self.judge.action_encoder_judge(prev_relation, prev_relation) #Dummy start argument for t = 0.

        all_loss = []  # list of loss tensors each [B,]
        all_logits = []  # list of actions each [B,]
        action_idx = []
        all_temp_logits_judge = []
        arguments_representations = []
        all_rewards_agents = []
        all_rewards_before_baseline = []
        prev_state_agent_1, prev_state_agent_2 = get_prev_state_agents()

        for t in range(T):

            next_possible_relations = candidate_relation_sequence[t]
            next_possible_entities = candidate_entity_sequence[t]
            current_entities_t = current_entities[t]

            which_agent_t = which_agent[t]
            loss, prev_state_agent_1, prev_state_agent_2, logits, idx, chosen_relation= \
                self.step(next_possible_relations, next_possible_entities,
                          prev_state_agent_1, prev_state_agent_2, prev_relation,
                          current_entities_t, range_arr=range_arr,
                          which_agent=which_agent_t, random_flag=random_flag)

            all_loss.append(loss)
            all_logits.append(logits)
            action_idx.append(idx)
            prev_relation = chosen_relation

            argument = self.judge.extend_argument(argument, tf.constant(t,dtype=tf.int32), idx, candidate_relation_sequence[t], candidate_entity_sequence[t],
                                                  range_arr)

            # If it is not the last part of an argument
            if t % self.path_length != (self.path_length - 1):
                all_temp_logits_judge.append(tf.zeros([self.batch_size,1]))
                temp_rewards = tf.zeros([self.batch_size,1])
                all_rewards_before_baseline.append(temp_rewards)
                all_rewards_agents.append(temp_rewards)
            else:
                logits_judge, rep_argu = self.judge.classify_argument(argument)
                rewards = tf.nn.sigmoid(logits_judge)
                all_temp_logits_judge.append(logits_judge)
                arguments_representations.append(rep_argu)
                all_rewards_before_baseline.append(rewards)
                if self.custom_baseline: #TODO: Remove custom baseline?
                    no_op_arg = self.judge.action_encoder_judge(prev_relation, prev_relation)  # Dummy start argument for t = 0.
                    for i in range(self.path_length):
                        no_op_arg = self.judge.extend_argument(no_op_arg, tf.constant(i,dtype=tf.int32), tf.zeros_like(idx),
                                                               candidate_relation_sequence[0], candidate_entity_sequence[0],
                                                               range_arr)
                    no_op_logits, rep_argu = self.judge.classify_argument(no_op_arg)
                    rewards_no_op = tf.nn.sigmoid(no_op_logits)
                    all_rewards_agents.append(rewards - rewards_no_op)
                else:
                    all_rewards_agents.append(rewards)


        loss_judge, final_logit_judge = self.judge.final_loss(arguments_representations)

        return loss_judge, final_logit_judge, all_temp_logits_judge, all_loss, all_logits, action_idx, all_rewards_agents, all_rewards_before_baseline
