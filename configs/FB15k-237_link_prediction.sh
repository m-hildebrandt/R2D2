data_input_dir="datasets/FB15k-237_link_prediction/"
base_output_dir="output/FB15k-237_link_prediction/"
vocab_dir="datasets/FB15k-237_link_prediction/vocab"
seed=42
is_use_fixed_false_facts=0
use_entity_embeddings=1
train_entity_embeddings=1
train_relation_embeddings=1
total_iterations=2200
eval_every=2200
save_debate_every=220
train_judge_every=200
rounds_sub_training=1800
path_length="2"
number_arguments=3
batch_size=64
embedding_size=64
hidden_size="64"
layers_judge=3
layers_agent=2
beta=0.1
Lambda=0.02