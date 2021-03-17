# XAI-Sentiment-Analysis

Explainable AI Project for sentiment analysis of tweets.


## Prerequisites
* Python3.8
* pip

## Setting Up the Project
### Clone the Project:
```sh
git clone git@github.com:gravlaks/XAI-Sentiment-Analysis.git
```


### Set Up Virtual Environment
Windows:
```sh
cd XAI-Sentiment-Analysis
python -m venv venv
source venv/Scripts/activate # Must be done every session
deactivate # If you want to deactivate venv, should be activated when working with the project
```

Linux:
```sh
cd XAI-Sentiment-Analysis
python3.8 -m venv venv
source venv/bin/activate # Must be done every session
deactivate # If you want to deactivate venv, should be activated when working with the project
```

### Install and Set Up Required Packages
```sh
pip install -r requirements.txt   # Important that venv is activated
python setup.py
```


#### Problems with NLTK
If setup.py does not work for any reason, try:
```
import nltk
nltk.download()
```
for a GUI where download of the packages is easier.

### Add the Necessary Data
Add a new directory named data to your project:
```sh
cd XAI-Sentiment-Analysis
mkdir data
```
[Download dataset](https://www.kaggle.com/kazanova/sentiment140) and place in `/data`  
[Download glove](https://www.kaggle.com/watts2/glove6b50dtxt) and place in `/data`



## Run in Jupyter Notebook
```sh
python -m ipykernel install --user --name=venv
jupyter notebook
```
* Open pipeline.ipynb in the jupyter tab in your web browser
* Set kernel to venv


# Future Development

## Adding Packages

Add package:
```sh
# Important that venv is activated before you add a new package
pip install -r requirements.txt #To ensure you have all current packages
pip install <name_of_module>
pip freeze > requirements.txt
```


