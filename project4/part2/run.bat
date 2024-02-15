@echo off

REM Setup a virtual env at the root directory and install dependencies
REM Or Python will try to use global dependencies
REM Uncomment below line activate env
REM call .\env\Scripts\activate

REM Uncomment below line to install dependencies (preferably within virtual env)
REM pip install -r requirements.txt

REM Running Scripts
REM extracted dataset direcory from dataset.zip must be present at the root dir
python subpart_1.py 1> ./subpart1_output.txt
python subpart_3.py 1> ./subpart3_output_1.txt
python subpart_3.py 1> ./subpart3_output_2.txt
python subpart_3.py 1> ./subpart3_output_3.txt
python subpart_3.py 1> ./subpart3_output_4.txt
python subpart_3.py 1> ./subpart3_output_5.txt