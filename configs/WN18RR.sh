data_input_dir="datasets/WN18RR/"
base_output_dir="output/WN18RR/"
vocab_dir="datasets/WN18RR/vocab"
seed=42
is_use_fixed_false_facts=1
use_entity_embeddings=1
train_entity_embeddings=1
train_relation_embeddings=1
total_iterations=6000
eval_every=6000
save_debate_every=1000
train_judge_every=200
rounds_sub_training=4000
path_length="3"
number_arguments="3"
batch_size=64
embedding_size=64
hidden_size="64"
layers_judge=4
layers_agent=1
beta=0.1
Lambda=0.02