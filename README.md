# CS-598--Forest_Sensing


# One more file yet to be added- generating github friendly version


Steps to Tree Segementation Code:
1. Python 3.6 version recommended
2. Navigate to the `tree segemenation folder`.
3. Run `train.py`
4. To visualize the segmentation from the pretrained model at an inference stage , run the `inference_script.ipynb`.



Steps to run the tree species classification:
1. Python 3.6 version recommended
2. Navigate to the `tree species classifcation folder`.
3. Unzip the files in `data` folder and save them in the same name `train` and `test`. 
4. Ensure that `train` and `test` are the folder names.
5. Run the `.ipynb` file cell by cell to see the results


Steps to run planning:
1. Python libararies os, numpy, scipy, pandas, seaborn, matplotlib, and pickle required. 
2. Navigate to `planning/src`.
3. Run `main.py` to run experiments. This will create `planning/outputs` and will fill with .png maps and .pkl data files.
4. Run `process_results.py` to process the results. This saves files `planning/outputs/all_data_${i}.pkl` with processed data in the form of numpy arrays. 
5. To unzip and process data, run `make_plots.py`. This file reads outputs from step 4, and creates one large dataframe. Results can be sorted, sifted, etc as desired by the user. The output plots for the current file are presented in our paper. 

`planning/inputs` contains the Google Maps inputs for the planning part of the project, as well as the 200ft-to-pixels ratio. 
