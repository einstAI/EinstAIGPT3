### A LCE for Mumford Grammar


Source code of Feature Encoding and Model Training for cardinality and cost estimation based on PostgreSQL Execution Plans.
The tree-structured model can generate representations for sub-plans which could be used by other query processing tasks like Join Reordering,
Materialized View and even Index Selection.

## 1. Feature Encoding

### 1.1. Cardinality Estimation



### Unitest
```bash
export PYTHONPATH=code/
python -m unittest code.test.test_feature_extraction.TestFeatureExtraction
python -m unittest code.test.test_feature_encoding.TestFeatureEncoding
python -m unittest code.test.test_training.TestTraining
```

### Test Data
We offer the test data including sampled Execution Plans from PostgreSQL,
IMDB datasets, Statistics of Minimum and Maximum number of each columns.
However, the pre-trained dictionary for string keywords is too large,
users can generate it by using the code in token_embedding
module which is hard-coded for JOB workload. 


## 2. Model Training
We use the tree-structured model to encode the sub-plans and use the
encoded representations to predict the cardinality and cost of the sub-plans.
The model is trained by using the sampled Execution Plans from PostgreSQL.

The model is implemented by PyTorch and the training process is implemented
by PyTorch Lightning. 

[click here for files](https://cloud.tsinghua.edu.cn/f/930a0ab8546b407a826b/?dl=1)  
Password: end2endsun

### 2.1. Cardinality Estimation
Note that the cardinality estimation is based on the cardinality of the sub-plans.
The cardinality of the sub-plans is estimated by using the cardinality of the whole plan.

### 2.2. Cost Estimation
The cost estimation is based on the cost of the sub-plans. 

### 2.3. Training
```bash
export PYTHONPATH=code/
python code/train.py --config config/config.yaml
```

### 2.4. Testing
```bash
export PYTHONPATH=code/
python code/test.py --config config/config.yaml
```

### 2.5. Inference
```bash
export PYTHONPATH=code/
python code/inference.py --config config/config.yaml
```

## 3. Model Inference
We provide the trained model for cardinality and cost estimation.
The trained model is implemented by PyTorch and the inference process is implemented
by PyTorch Lightning.

### 3.1. Cardinality Estimation
```bash
export PYTHONPATH=code/
python code/inference.py --config config/config.yaml --mode cardinality
```

### Citation
```
@article{sun2019endtoend,
  title={An End-to-End Learning-based Cost Estimator},
  author={Ji Sun and Guoliang Li},
  journal={arXiv preprint arXiv:1906.02560},
  year={2019}
}
```

### Contact
If you have any issue, feel free to post on [Project](https://github.com/greatji/Learning-based-cost-estimator).
