# Aspect Terms Extraction
Evaluated two types of methodologies WITHOUT Deep Learning: word vector and semantic features

## How to run
pip install -r requirements.txt\
python main.py

## Requirements:
Python 3.5 (other version should be fine, not tested though)
* numpy
* pandas
* scikit-learn
* imbalanced-learn
* py-xgboost
* python-crfsuite
* matplotlib
* nltk
* lxml

## Module Structure:
![alt text](https://github.com/showei/AspectExtractor/blob/master/module_structure.jpg "Logo Title Text 1")



## Module descriptions:
### PreProcessing:
Include XML Parser, Data cleaning, formatting
### Models:
Include classifier Benchmarker for Vectorized Features, and CRFClassifier
### PostProcessing:
Convert Extended BIO to regular BIO tag format, as well as convolution format to sentence format.

## License:
No License, Free to Use :)