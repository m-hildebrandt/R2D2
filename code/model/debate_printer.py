import numpy as np

class Debate_Printer:
    '''
    Class for handling the printing of the debates.
    '''

    def __init__(self, output_dir, grapher, num_rollouts, is_append=False, is_ranking=False):
        '''
        Initialzies a debate printer

        :param output_dir: str. Directory path where the results should be printed.
        :param grapher: Grapher. Grapher of the environment that the debates use.
        :param num_rollouts: int. Number of rollouts per query triple.
        :param is_append: bool. Flag to choose whether debates should be appended to the output file or the output file
        should be overwritten.
        :param is_ranking: bool. Flag to differentiate between link prediction and fact prediction. If True, that means
        the mode is link prediction and only the one true query is printed. This is to avoid writing large files, since
        the ranking can have over 200 entries.
        '''

        self.output_dir = output_dir
        self.grapher = grapher
        self.is_ranking = is_ranking
        self.num_rollouts = num_rollouts
        self.is_append = is_append
        self.debate_list = []

    def get_action_rel_ent(self,action_idx, state):
        '''
        Returns the human readable names of the actions' entities and relations as given by the action id.

        :param action_idx: Numpy array, [Batch_size]. Array with the id of the chosen actions. Values between 0 and max_num_actions.
        :param state: Dict. State of the environment. This has to be the state BEFORE updating with the selected actions,
        i.e. before calling Episode..
        :return: Tuple of two numpy arrays, each [Batch_size]. Each tuple contains the names for the selected relations
        and entities, respectively, for each debate in the batch.
        '''

        rel = np.copy(state['next_relations'][np.arange(state['next_relations'].shape[0]), action_idx])
        ent = np.copy(state['next_entities'][np.arange(state['next_entities'].shape[0]), action_idx])

        v_entity_string = np.vectorize(self.grapher.get_entity_string)
        v_relation_string = np.vectorize(self.grapher.get_relation_string)

        rel_string = v_relation_string(rel)
        ent_string = v_entity_string(ent)

        return rel_string, ent_string


    def create_debates(self, subject, predicate, object, label):
        '''
        Creates the Debates for the episode using the query information.

        The debates are added to self.debate_list.
        When created, the debates only have query information but no arguments (yet).

        :param subject: Numpy array [Batch_size]. Array with the ids of the queries' subjects.
        :param predicate: Numpy array [Batch_size]. Array with the ids of the queries' predicates.
        :param object: Numpy array [Batch_size]. Array with the ids of the queries' objects.
        :param label: Numpy array [Batch_size,1]. Array wit the labels of the queries.
        :return: None.
        '''

        v_entity_string = np.vectorize(self.grapher.get_entity_string)
        v_relation_string = np.vectorize(self.grapher.get_relation_string)

        sub_string = v_entity_string(subject)
        pred_string = v_relation_string(predicate)
        obj_string = v_entity_string(object)
        label = np.reshape(label,[-1])
        param_array = np.stack([sub_string,pred_string,obj_string,label],axis=1)

        self.debate_list = list(map(lambda params: Debate(*params), param_array))



    def create_arguments(self,rel_string_list, ent_string_list, confidence, is_pro):
        '''
        Creates the Arguments given by the list of relation and entities names and adds them to the corresponding Debates.

        :param rel_string_list: List of numpy arrays, each [Batch_size]. Each array contains the names of the relations
        selected at that time-step
        :param ent_string_list: List of numpy arrays, each [Batch_size]. Each array contains the names of the entities
        selected at that time-step.
        :param confidence: Numpy array, [Batch_size,1]. Array with the final confidence of the judge for the arguments.
        :param is_pro: Bool. Flag to differentiate between arguments presented by the pro agent and those presented by
        the con agent.
        :return: None.
        '''

        # action_idx_list == List of path_length with arrays [batch_length
        rel_string_list = np.stack(rel_string_list,axis=1)
        rel_string_list = list(rel_string_list) # List of batch_length size, with arrays [path_length]

        ent_string_list= np.stack(ent_string_list,axis=1)
        ent_string_list= list(ent_string_list) # List of batch_length size, with arrays [path_length]

        confidence = np.reshape(confidence,[-1])

        arguments = map(lambda rel_ent_tuple: Argument(*rel_ent_tuple,is_pro), zip(rel_string_list, ent_string_list, confidence))

        self.add_arguments(arguments)


    def add_arguments(self, arguments):
        '''
        Adds a sequence of arguments to the corresponding debates.

        :param arguments: Ordered iterable with the arguments to add for each debate.
        :return: None.
        '''

        for debate, argument in zip(self.debate_list, arguments):
            debate.add_argument(argument)


    def set_debates_final_logit(self, logits):
        '''
        Sets the final logits for the debates.

        :param logits: Numpy array, [Batch_size,1]. Array with the final logits for each debate in the batch.
        :return: None.
        '''

        logits = np.reshape(logits, [-1])
        for debate, logit in zip(self.debate_list,logits):
            debate.set_final_logit(logit)


    def create_best_debates(self):
        '''
        Creates the best debates for the episode.

        Best debates are the debates that take the 'best' arguments across all rollouts for the same query. Best arguments
        are those where the confidence was the highest (for the pro agent) or the lowest (for the con agent). The best
        debates do not have yet the final logits.

        Internally, this method clears self.debate_list and repopulates it with debates containing the first arguments
        in a list after sorting them according to their confidence. The chosen arguments are kept as diverse
        as possible. Concretely, this means that, if possible, no two arguments sharing the initial relation and target
        entity are selected.

        :return: None
        '''

        new_debate_list = []
        rollout_debate = set()

        for i, debate in enumerate(self.debate_list):
            rollout_debate.add(debate)
            if (i % self.num_rollouts) == self.num_rollouts - 1:
                new_debate_list.append(rollout_debate)
                rollout_debate = set()

        self.debate_list.clear()
        for debate_set in new_debate_list:
            temp_debate = next(iter(debate_set))
            best_debate = temp_debate.copy_debate_query()
            num_arguments = len(temp_debate.pro_arguments)
            all_pro_args = set()
            all_con_args = set()
            for debate in debate_set:
                all_pro_args.update(debate.pro_arguments)
                all_con_args.update(debate.con_arguments)

            sorted_pro_args = sorted(all_pro_args, key=lambda arg: arg.get_confidence(), reverse=True)
            sorted_con_args = sorted(all_con_args, key=lambda arg: -arg.get_confidence(), reverse=True)

            best_pro_args = []
            best_con_args = []

            # Fill the debates with the best, not-similar arguments possible.
            for pro_arg in sorted_pro_args:
                if not pro_arg.is_similar_set(best_pro_args):
                    best_pro_args.append(pro_arg)
                    best_debate.add_argument(pro_arg)
                if len(best_pro_args) == num_arguments:
                    break

            for con_arg in sorted_con_args:
                if not con_arg.is_similar_set(best_con_args):
                    best_con_args.append(con_arg)
                    best_debate.add_argument(con_arg)
                if len(best_con_args) == num_arguments:
                    break

            # If there are not enough diverse arguments, then retravel the sorted arguments and allow to append similar ones.
            if len(best_pro_args) != num_arguments:
                for pro_arg in sorted_pro_args:
                    if not pro_arg.is_in_set(best_pro_args):
                        best_pro_args.append(pro_arg)
                        best_debate.add_argument(pro_arg)
                    if len(best_pro_args) == num_arguments:
                        break

            if len(best_con_args) != num_arguments:
                for con_arg in sorted_con_args:
                    if not con_arg.is_in_set(best_con_args):
                        best_con_args.append(con_arg)
                        best_debate.add_argument(con_arg)
                    if len(best_con_args) == num_arguments:
                        break


            self.debate_list.append(best_debate)


    def write(self, file_name):
        '''
        Writes a representation of all debates found in self.debate_list to self.output_dir as a {file_name}.txt.

        Depending on the value of self.append, previous files are overwritten or appended.

        :param file_name: str. Name of the output file.
        :return: None.
        '''

        ret = ''
        for i, debate in enumerate(self.debate_list):
            ret += '---- DEBATE {} ----'.format(i)
            ret += '\n\t {}'.format(debate.to_string(self.grapher.get_placeholder_string()))
            if self.is_ranking:
                break

        file_name = self.output_dir + '/{}'.format(file_name)
        if self.is_append:
            with open(file_name,'a') as file:
                file.write(ret)
        else:
            with open(file_name,'w') as file:
                file.write(ret)


class Debate:
    '''
    Class representing a Debate.

    A debate has a query, a label, a number of pro and con arguments and a final value for the outcome of the debate.
    '''

    def __init__(self, sub, pred, obj, label):
        '''
        Initialzies a Debate.

        A Debate is initalized only by the query and label. The arguments and final outcome have to be set latter.

        :param sub: str. Query's subject.
        :param pred: str. Query's predicate.
        :param obj: Query's object.
        :param label: Query's label.
        '''

        self.subject = sub
        self.predicate = pred
        self.object = obj
        self.label = label
        self.pro_arguments = []
        self.con_arguments = []
        self.final_logit = None


    def add_argument(self,argument):
        '''
        Adds an argument to either the pro arguments or the con arguments of this debate.

        :param argument: Argument. Argument to add.
        :return: None.
        '''

        if argument.get_is_pro():
            self.pro_arguments.append(argument)
        else:
            self.con_arguments.append(argument)


    def to_string(self, sub_placeholder):
        '''
        Returns a string representation of the debate. This includes the query information, final outcome and pro and con
        arguments of the debate.

        :param sub_placeholder: str. Placeholder string to replace the query's subject.
        :return: str. String representation of the debate.
        '''

        ret = 'Query: {}\t{}\t{}'.format(self.subject, self.predicate, self.object)
        ret += '\n\t Query is {}'.format('True' if int(self.label) else 'False')
        ret += '\n\t Final logit: [{}]'.format(self.final_logit)
        for i in range(len(self.pro_arguments)):
            ret += '\n\n\t Pro argument {} ({}):\n'.format(i, 'Correct' if self.label else 'Incorrect')
            ret += '\t\t\t' + self.pro_arguments[i].to_string(self.subject, sub_placeholder)

            ret += '\n\n\t Con argument {} ({}):\n'.format(i, 'Correct' if not self.label else 'Incorrect')
            ret += '\t\t\t' + self.con_arguments[i].to_string(self.subject, sub_placeholder)

        return ret


    def set_final_logit(self, final_logit):
        '''
        Setter for final_logit.

        :param final_logit: float. Final logit/outcome of the debate.
        :return: None.
        '''

        self.final_logit = final_logit


    def copy_debate_query(self):
        '''
        Returns a copy of the debate.

        The copy only includes the query's information and label, not the final outcome or arguments.

        :return: Debate. Copy of the debate.
        '''
        return Debate(self.subject, self.predicate, self.object, self.label)


class Argument:
    '''
    Class representing an Argument.

    An argument is a sequence of relation and entity pairs that represent the path of the argument. This does not include
    the query, in particular not the query's subject. Additionally, an argument has the confidence or final value of the
    argument and a flag to differentiate between the argument being selected by the pro or the con agent.
    '''

    def __init__(self,rel, ent, confidence, is_pro):
        '''
        Initializes an argument.

        :param rel: Numpy array [path_length]. Array with the human readable names of all relations in the argument.
        :param ent: Numpy array [path_length]. Array with the human readable names of all entities in the argument.
        :param confidence: float. Confidence of the judge for the argument.
        :param is_pro: bool. Flag to signal whether the argument was presented by the pro or the con agent.
        '''
        self.rel_array = rel
        self.ent_array = ent
        self.confidence = confidence #float
        self.is_pro = is_pro

    def get_is_pro(self):
        '''
        Getter for self.is_pro.

        :return: bool. Value of self.is_pro.
        '''

        return self.is_pro

    def get_confidence(self):
        '''
        Getter for self.confidence.

        :return: float. Value of self.confidence.
        '''

        return self.confidence

    def is_similar(self,arg):
        '''
        Returns if this argument is similar to another argument.

        Two arguments are similar if they share the same first relation and firt entity.

        :param arg: Argument. Argument to compare this argument to.
        :return: bool. True if the arguments are similar. False otherwise.
        '''
        return self.rel_array[0] == arg.rel_array[0] and self.ent_array[0] == arg.ent_array[0]


    def is_similar_set(arg, used_args):
        '''
        Returns if this argument is similar to any argument in a collection of arguments.

        :param used_args: Iterable. Iterable of arguments.
        :return: bool. True if this argument is similar to any argument in used_args. False otherwise.
        '''

        for used_arg in used_args:
            if arg.is_similar(used_arg):
                return True

        return False

    def is_in_set(self, used_args):
        '''
        Returns if this argument is in a collection of arguments.


        :param used_args: Iterable. Iterable of arguments.
        :return: bool. True if this argument is in used_args. False otherwise.
        '''

        for used_arg in used_args:
            if self == used_arg:
                return True

        return False


    def to_string(self, subject, sub_placeholder):
        '''
        Returns a string representation of the argument. This includes the sequence of chosen relations and target entities,
        as well as the final confidence of the judge for the argument.

        :param subject: str. Query's subject of the argument.
        :param sub_placeholder: str. Placeholder string to replace the query's subject.
        :return: str. String representation of the argument.
        '''

        ret = sub_placeholder

        for rel, ent in zip(self.rel_array, self.ent_array):
            ret += ' -> {} '.format(rel)
            ret += ' -> {} '.format(ent if ent != subject else sub_placeholder)
        ret += '\n\t\t\tConfidence: [{}]\n'.format(self.confidence)

        return ret








