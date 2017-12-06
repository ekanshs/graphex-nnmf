# Sparse Exchangeable Non-Negative Matrix Factorization

## About

This package implements coordinate-ascent variational inference for sparse exchangeable bipartite graph model described in the article *Exchangeable Modelling of Relational Data: Checking Sparsity, Data Splitting, and Sparse Exchangeable Poisson Matrix Factorization*


## Requirements

* Python in [2.7.x]
* Tensorflow in [+1.1.x]
* Tensorflow w/ GPU in [+1.1.x] for GPU mode
* 64-bit architecture

## Scripts
We provide two scripts, 1. For simulating test dataset, 2. to run the model on a given dataset:

#### data/simulate_data.py
Simulates a dataset with given parameter setting; Splits the data into Train, Lookup and Holdout set

For usage, run:
```bash
cd scripts
python simulate_data.py -h
```

#### run_script.py
This script runs the model on a given dataset.
Requires three files in the data directory: train.pkl, test_lookup.pkl, test_holdout.pkl
For the format in which data should be stored, check data/simulate_data.py

For usage, run: 
```bash
cd scripts
python run_script.py -h
```

## Contributors

* Ekansh Sharma 
* Victor Veitch

## References

* [Veitch, Sharma, Naulet, and Roy, *Exchangeable Modelling of Relational Data: Checking Sparsity, Data Splitting, and Sparse Exchangeable Poisson Matrix Factorization*, 2017](http://victorveitch.com/assets/pdfs/deanon.pdf)
* [Naulet, Sharma, Veitch, and Roy, *An estimator for the tail-index of graphex processes*, 2017](https://arxiv.org/pdf/1712.01745.pdf)