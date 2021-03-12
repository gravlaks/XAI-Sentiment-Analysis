# XAI-Sentiment-Analysis

Explainable AI Project for sentiment analysis of tweets.


## Prerequisites
* Python3.8
* pip

## Setting up the Project
Clone the project:
```sh
git clone git@github.com:gravlaks/XAI-Sentiment-Analysis.git
```

Set up Virtual Environment
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

Install and setup required packages
Windows:
```sh
pip install -r requirements.txt   # Important that venv is activated
python setup.py
```

Linux:
```sh
pip install -r requirements.txt   # Important that venv is activated
./setup.py
```

Prepare the necessary data
Add a new directory named data to your project:
```sh
mkdir data
```
[Download dataset](https://www.kaggle.com/kazanova/sentiment140) and place in `/data`
[Download glove](https://www.kaggle.com/watts2/glove6b50dtxt) and place in `/data`


## Run Preprocessing Script

The preprocessing script takes in the data set and (optionally) produces an output file from it.

```shellscript
# Linux

./pre.py -i <input data set> -o <output file>
# Same as:
./pre.py --input <input data set> --output <output file>

# Windows
python pre.py -i <input data set> -o <output file>
```

## Load Preprocessed Output File

The preprocessed output file can be loaded easily by using the `load_data` function from `parse`.

```python
from parse import load_data

df = load_data('data/preprocessed-data-set.csv')
print(df.columns)
```

## Installing and Adding Packages
Installing required packages:
```sh
pip install -r requirements.txt
```

Add package:
```sh
# Important that venv is activated before you add a new package
pip install <name_of_module>
pip freeze > requirements.txt
```
