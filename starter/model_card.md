# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
* Made by: manualrg
* on: 2024-01-01
* version: 0.0.1
* model type: Supervised Learning. Binary Classification

### Model info:
* target: "salary"
* taret levels: "<=50K", ">50K"
* hiperarameter tunning
    * C: 1
* feature engineering:
    * ohe categorial features


## Intended Use
Academic research on model lifecycle tools

## Training Data
* dataset: https://archive.ics.uci.edu/dataset/20/census+income
* description: Prediction task is to determine whether a person makes over 50K a year. 
* downloaded on: 2024-01-01
* Cross validation results: model/cs_summary.csv

## Evaluation Data
* dataset: 20% split from Training Data
* Sliced evaluation on sex: model/metrics_by_slice.csv

## Metrics
* f1-score: 0.3715

## Ethical Considerations
As an academic trial focused on stablishing a full model life cycle, there are no major concerns on Ethical Considerations.
On an actual research or census or demographic data, other results will be more worthly to explore

## Caveats and Recommendations
From the point of view of a MLOps reserach:
* Tests are implementeted as an example, trying to fit the philosofy of unit testing in the best way
    * test_process_data: Ensure that changes does not break the data interface
    * test_inference: idem
    * test_train_model: Runs the training function in a toy dataset thatn can be run on a light machine to check any issues with the model. May be much more usefull with deep neural nets
* Source code structure is not optimized to be a software project, but to be an academic project focused on deployment
