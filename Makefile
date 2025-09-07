
# Define the variables for the Python interpreter
VENV = venv
PYTHON = $(VENV)/bin/python3
PIP = $(VENV)/bin/pip

# Declare phony target with no prerequisites
# .PHONY: all
# all: run
.PHONY: run clean test

# Creating first rule to run the app in the virtual environment:
run: venv/bin/activate
	$(PYTHON) src/app.py


# Run the neural network in the virtual environment with the app as prerequisite
run_neural_network: venv/bin/activate, run
	$(PYTHON) src/neural_layers.py

# Install dependencies
#.PHONY: install
#install:
#	$(PYTHON) -m pip install -r requirements.txt

# Creating virtual environment with updated dependencies
venv/bin/activate: requirements.txt
	python3 -m venv venv
	$(PIP) install -r requirements.txt

# Run tests
test: venv/bin/activate
	$(PYTHON) -m unittest discover -s tests

# Clean up .pyc files and refresh 
clean:
#	find . -name "*.pyc" -delete
    rm -rf __pycache__
	rm -rf venv
	