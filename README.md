# DS340W-SP25_Plant-Disease-Classification
DS340W SP25 Plant disease classification project using robust training data and lightweight deep neural architectures.

The configs and the models were changed, with some testing done to start the transfer learning process and familiarize ourselves with the feature extractions mentioned in parent paper 1.


If dataset is preprocessed, skip ahead to step 4

1. Run merge_and_standardize_datasets.py, with the datasets you want in same format, with same variable names. 2. Modify the .py file to include additional datasets if required.

3. Then, run dataset_preparation.py, to map the dataframes.


4. Then, modify config.json to include appropriate training parameters and data directory.

5. Then, run train.py.
When finished training, model training results will be in the directory specified previously.

6. Finally, evaluate model performance by running evaluate.py.