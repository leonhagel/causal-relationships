SHELL := /bin/bash
ENV?=./.env
PORT?=8888
PYTHON?=python3.9
REQUIREMENTS?="setup/requirements.txt"
RSCRIPT?="Rscript"
MODEL?=

help:
	@echo "Make targets:"
	@echo "  build              setup python virtual environment and install R dependencies (use: make build PY=/path/to/python to specify the python executable')"
	@echo "  clean              remove *.pyc files and __pycache__ directory"
	@echo "  distclean          clean + remove virtual environment and cache"	
	@echo "  data-cleaning      perform data cleaning"	
	@echo "  feature-creation   perfrom feature creation"
	@echo "  grid-search        perform grid-search on all models (use: make grid-seach MODEL=model-name to perform the grid search on single model)"
	@echo "  model-training     train all models (use: make model-training MODEL=model-name to train a single model)"
	@echo "  lab                run jupyter lab (default port: $(PORT))"
	@echo "Check the Makefile for details"

build: build-py build-r
	mkdir -p cache data out; \
       	mkdir -p out/grid-search out/img out/markov-blankets out/predictions out/selected-features out/training-stats; \
	source $(ENV)/bin/activate; \
	cd src; \
	python utils/config.py "Rscript" $(RSCRIPT)

build-py: 
	virtualenv --python=$(PYTHON) $(ENV)
	source $(ENV)/bin/activate; \
	python -m pip install --upgrade pip; \
	python -m pip install -r $(REQUIREMENTS);

build-r:
	$(RSCRIPT) setup/requirements.R 

clean:
	find . -name '*.pyc' -delete
	find . -name '__pycache__' -type d | xargs rm -fr
	find . -name '.ipynb_checkpoints' -type d | xargs rm -fr

distclean: clean
	rm -rf cache
	rm -rf $(ENV)

data-cleaning:
	source $(ENV)/bin/activate; \
	cd src; \
	python data_cleaning.py;

feature-creation:
	source $(ENV)/bin/activate; \
	cd src; \
	python feature_creation.py;

grid-search:
ifdef MODEL
	source $(ENV)/bin/activate; \
	cd src; \
	python grid-search/$(MODEL).py; 
else
	source $(ENV)/bin/activate; \
	cd src; \
	python grid-search/boosting.py; \
	python grid-search/castle.py; \
	python grid-search/causal-selection_ges.py; \
	python grid-search/causal-selection_lingam.py; \
	python grid-search/causal-selection_notears.py; \
	python grid-search/causal-selection_pc.py; \
	python grid-search/decision-tree.py; \
	python grid-search/logistic-regression.py; 
endif

model-training:
ifdef MODEL
	source $(ENV)/bin/activate; \
	cd src; \
	python model-training/$(MODEL).py; 
else
	source $(ENV)/bin/activate; \
	cd src; \
	python model-training/boosting.py; \
	python model-training/castle-regularization.py; \
	python model-training/causal-selection.py; \	
	python model-training/decision-tree.py; \
	python model-training/feature-selection.py; \
	python model-training/logistic-regression.py; 
endif

lab:
	source $(ENV)/bin/activate; \
	jupyter lab --port $(PORT)
