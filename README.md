# R2D2 - Reveal Relations using Debate Dynamics

Tensorflow implementation of the method described in [Reasoning on Knowledge Graphs using Debate Dynamics](https://arxiv.org/abs/2001.00461)

R2D2 consists of a judge and two RL agents, a 'pro' and a 'con' agent. The agents are trained to mine paths in the KG that
either support an statement/thesis (pro agent) or oppose it (con agent). The paths are presented to the judge, which makes a 
prediction regarding the correctness of the thesis, similar to a judge deciding the outcome of a debate using arguments
presented by the parties.

<h2> Credits</h2>

This implementation of R2D2 is based on [Shehzaad Dhuliawala's repository](https://github.com/shehzaadzd/MINERVA) which contains the code for the paper [Go for a Walk and Arrive at the Answer - Reasoning over Paths in Knowledge Bases using Reinforcement Learning](https://arxiv.org/abs/1711.05851). 

<h2> How To Run </h2>

The dependencies are specified in [requirements.txt](requirements.txt). To run R2D2, create one of the default config files or create your own. For an explanation of each hyperparameter refer
to the README file in the configs folder. 

Then, run the command
```
./run.sh configs/${config_file}.sh
```


<h2> Data Format </h2>

<h5> Triple format </h5>

KG triples for R2D2 need to be written in the format ```subject predicate object```,
with tabs as separators. Furthermore, R2D2 uses inverse relations, so it is important to add the inverted triple for each
fact in the KG. Use  ```_``` before a predicate to signal the inverse relation, e.g. the inverted triple for
```Germany hasCapital Berlin``` is ```Berlin _hasCapital Germany```.

<h5> File format </h5>

Datasets for R2D2 should have the following files:
```
dataset
    ├── graph.txt
    ├── train.txt
    ├── dev.txt
    └── test.txt
```

Where:

```test.txt``` contains all test triples.

```dev.txt``` contains all validation triples.

```train.txt``` contains all train triples.

```graph.txt``` contains all remaining triples of the KG except for the inverses ```dev.txt``` and ```test.txt``` 
as well as all training triples. In other words, ```graph.txt``` is the whole KG minus the test and dev triples and their inverses. 
   
Finally, R2D2 needs two vocab files, one for the entities and one for the relations. You can create these using the 
```create_vocab.py``` file.

<h2> Citation </h2>

If you used this implementation, please cite:
```
@article{hildebrandt2020reasoning,
  title={Reasoning on Knowledge Graphs with Debate Dynamics},
  author={Hildebrandt, Marcel and Serna, Jorge Andres Quintero and Ma, Yunpu and Ringsquandl, Martin and Joblin, Mitchell and Tresp, Volker},
  journal={arXiv preprint arXiv:2001.00461},
  year={2020}
}
```

