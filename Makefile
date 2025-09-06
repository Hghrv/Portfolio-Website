# Define the Python interpreter
PYTHON = python3

# Default target
.PHONY: all
all: run

# Run the Python script
.PHONY: run
run:
	$(PYTHON) src/neural_layers.py

# Install dependencies
.PHONY: install
install:
	$(PYTHON) -m pip install -r requirements.txt

# Run tests
.PHONY: test
test:
	$(PYTHON) -m unittest discover -s tests

# Clean up .pyc files
.PHONY: clean
clean:
	find . -name "*.pyc" -delete

    rm -rf __pycache__
	