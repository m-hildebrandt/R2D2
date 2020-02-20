from __future__ import absolute_import
from __future__ import division
import argparse
import uuid
import os
import datetime
from pprint import pprint


def read_options():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_input_dir", default="", type=str)
    parser.add_argument("--base_output_dir", default='', type=str)
    parser.add_argument("--vocab_dir", default="", type=str)
    parser.add_argument("--load_model", default=0, type=int)
    parser.add_argument("--model_load_dir", default="", type=str)
    parser.add_argument("--pretrained_embeddings_path",default="0",type=str)
    parser.add_argument("--seed", default=None, type=int, nargs="+")
    parser.add_argument("--is_use_fixed_false_facts", default=0, type=int)
    parser.add_argument("--use_entity_embeddings", default=0, type=int)
    parser.add_argument("--train_entity_embeddings", default=0, type=int, nargs="+")
    parser.add_argument("--train_relation_embeddings", default=1, type=int, nargs="+")
    parser.add_argument("--total_iterations", default=2000, type=int, nargs="+")
    parser.add_argument("--eval_every", default=50, type=int)
    parser.add_argument("--save_debate_every", default=250, type=int)
    parser.add_argument("--train_judge_every",default=1,type=int,nargs="+")
    parser.add_argument("--rounds_sub_training", default=0, type=int, nargs="+")
    parser.add_argument("--max_num_actions", default=200, type=int, nargs="+")
    parser.add_argument("--path_length", default=3, type=int, nargs= "+")
    parser.add_argument("--number_arguments",default=3, type=int, nargs= "+")
    parser.add_argument("--batch_size", default=128, type=int, nargs= "+")
    parser.add_argument("--false_facts_train",default=1,type=int, nargs= "+")
    parser.add_argument("--num_rollouts", default=20, type=int, nargs= "+")
    parser.add_argument("--test_rollouts", default=100, type=int, nargs= "+")
    parser.add_argument("--embedding_size", default=50, type=int, nargs= "+")
    parser.add_argument("--hidden_size", default=50, type=int, nargs= "+")
    parser.add_argument("--layers_judge", default=1, type=int, nargs= "+")
    parser.add_argument("--layers_agent",default=6, type=int, nargs= "+")
    parser.add_argument("--learning_rate_judge", default=5e-4, type=float, nargs= "+")
    parser.add_argument("--learning_rate_agents", default=5e-4, type=float, nargs= "+")
    parser.add_argument("--gamma", default=1, type=float, nargs= "+")
    parser.add_argument("--grad_clip_norm", default=5, type=int, nargs="+")
    parser.add_argument("--beta", default=1e-2, type=float, nargs="+")
    parser.add_argument("--Lambda", default=0.0, type=float, nargs= "+")
    parser.add_argument("--custom_baseline", default=[0], type=int, nargs="+")


    try:
        parsed = vars(parser.parse_args())
    except IOError as msg:
        parser.error(str(msg))


    parsed['use_entity_embeddings'] = (parsed['use_entity_embeddings'] == 1)
    parsed['is_use_fixed_false_facts'] = (parsed['is_use_fixed_false_facts'] == 1)

    if parsed['pretrained_embeddings_path'] != '0':
        parsed['pretrained_embeddings_action'] = parsed['pretrained_embeddings_path'] +'/'+ "relations.p"
        parsed['pretrained_embeddings_entity'] = parsed['pretrained_embeddings_path'] +'/'+ "entities.p"
        parsed['pretrained_ent_dict'] = parsed['pretrained_embeddings_path'] +'/'+ "entities_dict.p"
        parsed['pretrained_rel_dict'] = parsed['pretrained_embeddings_path'] +'/'+ "relations_dict.p"
    else:
        parsed['pretrained_embeddings_action'] = '0'
        parsed['pretrained_embeddings_entity'] = '0'

    parsed['custom_baseline'] = list(map(lambda custom_baseline: custom_baseline == 1, parsed['custom_baseline']))
    # print and return
    maxLen = max([len(ii) for ii in parsed.keys()])
    fmtString = '\t%' + str(maxLen) + 's : %s'
    print('Arguments:')
    for keyPair in sorted(parsed.items()): print(fmtString % keyPair)
    return parsed
