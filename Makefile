
# Define the variables for the Python interpreter
VENV = venv
PYTHON = $(VENV)/bin/python3
PIP = $(VENV)/bin/pip

# Declare phony target with no prerequisites
# .PHONY: all
# all: run
.PHONY: run clean test pythonpath

# Creating first rule to run the app in the virtual environment:
run: venv/bin/activate
	$(PYTHON) src/app.py


# Run the neural network in the virtual environment with the app as prerequisite
run_neural_network: venv/bin/activate, run
	$(PYTHON) src/neural_layers.py

####################################################################################################

# Install dependencies
#.PHONY: install
#install:
#	$(PYTHON) -m pip install -r requirements.txt

# Creating virtual environment with updated dependencies
venv/bin/activate: requirements.txt
	python3 -m venv venv
	
	$(PIP) install -r requirements.txt
	$(PIP) install -r ./requirements.txt -t dependencies/python

# Setting the python path to the current working directory
pythonpath: venv/bin/activate
	export PYTHONPATH=$PWD

################################################################################################################
# Set Up Tests dependencies

## Install bandit
bandit:
	$(PIP) install bandit

## Install flake8
flake8:
	$(PIP) install flake8

## Install coverage
coverage:
	$(PIP) install coverage
	$(PIP) install pytest-cov

## Set up dev requirements (bandit, black)
dev-setup: bandit flake8 coverage

# Build / Run

## Run bandit
run-bandit:
	bandit -r src/

## Run flake8
run-flake8:
	flake8  ./src/*.py ./test/*.py

## Run the unit tests
unit-test:
	PYTHONPATH=${PYTHONPATH} pytest test/ -v

## Run the coverage check
check-coverage:
	PYTHONPATH=${PYTHONPATH} pytest --cov=src test/

## Run all checks
run-checks: run-bandit run-flake8 unit-test check-coverage


# Run second routine tests
test: venv/bin/activate
	$(PYTHON) -m unittest discover -s tests



# Clean up .pyc files and refresh 
clean:
#	find . -name "*.pyc" -delete
	rm -rf __pycache__
	rm -rf venv
	