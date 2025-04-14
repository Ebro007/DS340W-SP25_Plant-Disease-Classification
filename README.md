# DS340W-SP25_Plant-Disease-Classification
DS340W SP25 Plant disease classification project using robust training data and lightweight deep neural architectures.

The configs and the models were changed, with some testing done to start the transfer learning process and familiarize ourselves with the feature extractions mentioned in parent paper 1.


1. Modify merge_and_standardize_datasets.py to include paths to included datasets. There are siz datasets currently selected, but you can modify the 'datasets_list' to include a new name and path to the dataset of images separated by labeled folder for each class of disease. New classes can also be added by including them in the list of 'allowed_categories'.

2. Next, you must run bad_image.py to ensure all image paths are real and no images are corrupted.

3. Then, run dataset_preparation.py, to map the dataframes.

4. Then, modify config.json to include appropriate training parameters, an output path, and the input data directory ('./Tomato-Merged' by default).

5. Then, run train.py.
When finished training, model training results will be in the output directory.




The data in the repository is processed. 
If you want to run several models sequentially, then all the config.json files for the different models need to be modified and included in the run_multiple_configs.py file.

Otherwise, run train.py

    Depending on the version of tensorflow, you may need to adjust the callback model on line 50 of utils.py to save as .keras instead of .h5, and line 51 needs to become save_weights_only=False.

    May also need to modify the config.json file to include update dir path, model, and parameters.