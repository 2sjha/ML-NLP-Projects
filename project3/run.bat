@echo off

echo I've done this project in Python 3.11 but any version after 3.8 should work.
echo Current Python Version is 
python --version

REM Setup a virtual env at the root directory and install dependencies
REM Or Python will try to use global dependencies
REM Uncomment below line activate env
REM call .\env\Scripts\activate

REM Uncomment below line to install dependencies (preferably within virtual env)
REM pip install -r requirements.txt

echo.
echo Need to make sure all python scripts run from within src
cd src
cd
echo.

REM Running Scripts
REM netflix.zip must be present at the root dir
python all_models.py collab 1> ./../collab_output.txt
python all_models.py knn 1> ./../knn_output.txt
python all_models.py svm 1> ./../svm_output.txt
