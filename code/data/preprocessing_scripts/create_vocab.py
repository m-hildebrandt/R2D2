import json
import csv
import argparse
import os

def main():
    '''
    Creates the vocab files for a dataset.

    The vocab is in json format {'entity/relation': ID}. To both vocab some special entities/realtions are added that are
    used by R2D2.

    :return: None.
    '''

    root_dir = '../../../'
    vocab_dir = root_dir + 'datasets/FB15k_all_relations/vocab/'
    dir = root_dir + 'datasets/FB15k_all_relations/'

    if not os.path.isdir(vocab_dir):
        os.makedirs(vocab_dir)

    entity_vocab = {}
    relation_vocab = {}

    entity_vocab['PAD'] = len(entity_vocab)
    entity_vocab['UNK'] = len(entity_vocab)
    entity_vocab['QUERY_SUBJECT'] = len(entity_vocab)
    relation_vocab['PAD'] = len(relation_vocab)
    relation_vocab['DUMMY_START_RELATION'] = len(relation_vocab)
    relation_vocab['NO_OP'] = len(relation_vocab)
    relation_vocab['UNK'] = len(relation_vocab)
    relation_vocab['END_OF_ARGUMENT'] = len(relation_vocab) #TODO: Delete END_OF_ARGUMENT

    entity_counter = len(entity_vocab)
    relation_counter = len(relation_vocab)

    for f in ['train.txt', 'dev.txt', 'test.txt', 'graph.txt']:
        with open(dir + f) as raw_file:
            csv_file = csv.reader(raw_file, delimiter='\t')
            for line in csv_file:
                e1, r, e2 = line[0], line[1], line[2]
                if e1 not in entity_vocab:
                    entity_vocab[e1] = entity_counter
                    entity_counter += 1
                if e2 not in entity_vocab:
                    entity_vocab[e2] = entity_counter
                    entity_counter += 1
                if r not in relation_vocab:
                    relation_vocab[r] = relation_counter
                    relation_counter += 1

    with open(vocab_dir + 'entity_vocab.json', 'w') as fout:
        json.dump(entity_vocab, fout)

    with open(vocab_dir + 'relation_vocab.json', 'w') as fout:
        json.dump(relation_vocab, fout)

if __name__ == '__main__':
    main()
