#!/bin/bash

# Setup a virtual env at the root directory and install dependencies
# Or Python will try to use global dependencies
# Uncomment below line activate env
source ./env/bin/activate

# Uncomment below line if virtual env is setup and dependencies not installed
# pip install -r requirements.txt

# Running Scripts
# extracted dataset direcory from dataset.zip must be present at the root dir
python subpart_1.py 1> ./subpart1_output.txt
python subpart_3.py 1> ./subpart3_output_1.txt

# More runs to get average and variance of multiple runs as mentioned in the PDF
# python subpart_3.py 1> ./subpart3_output_2.txt
# python subpart_3.py 1> ./subpart3_output_3.txt
# python subpart_3.py 1> ./subpart3_output_4.txt
# python subpart_3.py 1> ./subpart3_output_5.txt