# causal-relationships
Applied Predictive Analytics | Summer 2022 | HU Berlin

This repository contains our work performed during the Applied Predictive Analytics seminar at HU Berlin. Our work analyzes the use of causal structure learning while predicting credit default. Our results can be found in our report (causal-relationships-in-credit-risk.ipynb).

**Authors:** Leon Hagel, Kieu-Long Huynh, Mehmet Oguzcan Kervanci, Peiqi Zhu 

### Repo Structure
```
.
├── appendix/                                    # analysis of grid search and model results
│   ├── grid-search.ipynb
│   └── model-evaluation.ipynb
├── data/                                        # kaggle data, cleaned data (not uploaded to github)
├── img/                                         # images used in the paper
├── out/                                         # model output
│   ├── grid-search/
│   ├── img/
│   ├── markov-blankets/
│   ├── predictions/
│   ├── selected-features/
│   └── training-stats/
├── setup/                                       # required files for setting up the environment 
├── src/
│   ├── CASTLE/                                  # CASTLE code provided by Kyono et. al (2020), slighly modified
│   ├── grid-search/                             # python scripts to perform grid-serach
│   ├── model-training/                          # python scripts to train the models
│   ├── models/                                  # modified model classes
│   ├── utils/                                   # required helper
│   ├── data-cleaning.py                         # python script used to perform data cleaning
│   └── feature-creation.py                      # python script used to perform feature creation
├── makefile
├── README.md
└── causal-relationships-in-credit-risk.ipynb    # Paper: Causal Relationship in predicting Credit Default
```

### Setting up the Environment (Python 3.9)
1. Clone the repo: `git clone https://github.com/leonhagel/causal-relationships.git`
1. Change the directory to the github repo: `cd causal-relationships`
1. Install virtualenv: `pip3.9 install virtualenv`
1. Build the virtual environment: `make build` 
    - to specify the python version used to create virtual environment: `make build PYTHON="/path/to/python"`, e.g. `make build PYTHON="/usr/bin/python3.9"`
    - to use a different requirements file use: `make build REQUIREMENTS="/path/to/new-requirements.txt"`, e.g. `make build REQUIREMENTS="setup/requirements-m1.txt"`
1. Start jupyter lab: `make lab`

### Reproducing our Results
1. Perform data cleaning: `make data-cleaning`
1. Perform feature creation: `make feature-creation`
1. Perform the grid search: `make grid-search`
1. Train the models: `make model-training`

### make targets
```
build              setup python virtual environment and install R dependencies (use: make build PY=/path/to/python to specify the python executable)
clean              remove *.pyc files and __pycache__ directory
distclean          clean + remove virtual environment and cache
data-cleaning      perform data cleaning	
feature-creation   perform feature creation
grid-search        perform grid-search on all models (use: make grid-seach MODEL=model-name to perform the grid search on single model)
model-training     train all models (use: make model-training MODEL=model-name to train a single model)
lab                run jupyter lab (default port: 8888)
```

### References
Kyono, T., Zhang, Y., & van der Schaar, M. (2020). CASTLE: regularization via auxiliary causal graph discovery. Advances in Neural Information Processing Systems, 33, 1501-1512. ([GitHub](https://github.com/trentkyono/CASTLE "CASTLE repository"))
