#!/bin/bash

# Check Python version
echo "I've done this project in Python 3.11 but any version >= 3.8 should work."
echo -n "Current Python Version is "
python --version

# Setup a virtual env at the root directory and install dependencies
# Or Python will try to use global dependencies
#  Uncomment below line activate env
# source ./env/bin/activate

# Uncomment below line if virtual env is setup and dependencies not installed
# pip install -r requirements.txt

echo -e "\nNeed to make sure all python scripts run from within src"
cd src
pwd
echo -e "\n"

# Running Scripts
# netflix.zip must be present at the root dir
python all_models.py collab 1> ./../collab_output.txt
python all_models.py knn 1> ./../knn_output.txt
python all_models.py svm 1> ./../svm_output.txt