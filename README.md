# A-Dynamic-Meta-Learning-Model-for-Time-Sensitive-Cold-Start-Recommendations

This is a Pytorch implementation of our model:

![Recommendation Framework](https://github.com/ritmininglab/A-Dynamic-Meta-Learning-Model-for-Time-Sensitive-Cold-Start-Recommendations/blob/main/final_model.png)

A novel dynamic recommendation model that focuses on users who have interactions in the past but turn relatively inactive recently i.e. _time-sensitive cold-start users_.
 

## Installation

1. Clone this repository.
   ```sh
   git clone https://github.com/ritmininglab/A-Dynamic-Meta-Learning-Model-for-Time-Sensitive-Cold-Start-Recommendations
   cd A-Dynamic-Meta-Learning-Model-for-Time-Sensitive-Cold-Start-Recommendations
   ```

2. Install the following dependencies. The code should run with Pytorch 1.3.1 and newer.
* Pytorch (1.3.x)
* python 3.5 or newer
* scikit-learn
* scipy
* numpy
* pickle

## Run

1. Go to each folders of datasets to run the corresponding experiments.
2. For example ```cd Netflix``` and 
3. Run ```python proposed_model.py``` for the Netflix dataset

## Base

This code is based on ```MeLU```
