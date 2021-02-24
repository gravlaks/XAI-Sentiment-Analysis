# XAI-Sentiment-Analysis

Explainable AI Project for sentiment analysis of tweets.

## Project setup

1. pip install -r requirements.txt
2. ./setup.py
3. [download dataset](https://www.kaggle.com/kazanova/sentiment140) and place in `/data`

## Set up virtual environment in Linux:

1. Install Python3.8 (if not installed)
2. Install pip (if not installed)
3. Run code below in terminal

```shellscript
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
