# Model Card

## Model Details
* Made by: manualrg
* on: 2024-01-01
* version: 0.0.1
* model type: Supervised Learning. Binary Classification

### Model info:
* target: "salary"
* target levels: "<=50K", ">50K"
* hiperparameter tunning
    * C: 1
* feature engineering:
    * ohe on categorical features


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

## Evaluation and Metrics
* Metric used is f1-score, as it is a trade-off between Precision and Recall,
in addition, it is a suitable metric in rare event problems. 
For a detailed implementation of the metric see: scikit-learn [user guide section on f1-score](https://scikit-learn.org/stable/modules/model_evaluation.html#precision-recall-f-measure-metrics)  and f1-score  [implementation](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html)
* f1-score value on test: 0.3715

## Ethical Considerations
As an academic trial focused on stablishing a full model life cycle, there are no major concerns on Ethical Considerations.

On an actual research on census or demographic data, results will be more worthly to explore with an ai-fairness framework. 

On this academic dataset, a data slicing evaluation is carried out on `sex` feature. This shows an over-representation of `Male` examples and a higher evaluation metric value on this slice. Further analysis should be carried out (leveraging and actual ai-fairnes framework) to analyze this insight 

## Caveats and Recommendations
From the point of view of a MLOps academic project:
* Tests are implementeted as an example, trying to fit the philosophy of unit testing in the best way
    * test_process_data: Ensure that changes does not break the data interface
    * test_inference: idem
    * test_train_model: Runs the training function in a toy dataset that can be run on a light machine to check any issues with the model. May be much more usefull with deep neural nets
* Source code structure is not optimized to be a software project, but to be an academic project focused on deployment
