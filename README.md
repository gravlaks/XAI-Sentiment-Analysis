# XAI-Sentiment-Analysis

Explainable AI Project for sentiment analysis of tweets.


## Prerequisites
* Python3.8
* pip

## Clone and Set Up Virtual Environment
Clone project:
```sh
git clone git@github.com:gravlaks/XAI-Sentiment-Analysis.git
```

Windows:
```sh
cd XAI-Sentiment-Analysis
python -m venv venv
source venv/Scripts/activate # Must be done every session

```

Linux:
```sh
cd XAI-Sentiment-Analysis
python3.8 -m venv venv
source venv/bin/activate # Must be done every session

```

Deactivate venv:
```sh
deactivate
```


## Project Setup

Windows:
```sh
pip install -r requirements.txt
python setup.py
mkdir data      # Place data shown belown in the /data

```

Linux:
```sh
pip install -r requirements.txt
./setup.py
mkdir data      # Place data shown belown in the /data

```

[Download dataset](https://www.kaggle.com/kazanova/sentiment140) and place in `/data`


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


3. Run code below in terminal

```sh
python3.8 -m venv venv
#Linux:
source venv/bin/activate #must be done every session
#Windows:
source venv/Scripts/activate
deactivate (to deactivate venv)
```

## Imported modules

Add package:

```shellscript
pip install <name_of_module>
pip freeze > requirements.txt
```

Import package other people have added:

```shellscript
pip install -r requirements.txt
```

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
