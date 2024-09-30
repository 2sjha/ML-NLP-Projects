#!/bin/bash

# Check Python version
echo "I've done this project in Python 3.11 but any version >= 3.8 should work."
echo -n "Current Python Version is "
python --version

# I'm doing all the work in Python 3.11 but Any python version >=3.8 should work
echo -e "\nNeed to make sure all python scripts run from within src"
cd src
pwd
echo -e "\n"

# Running Scripts
python all_models.py 2> ./../errors.txt