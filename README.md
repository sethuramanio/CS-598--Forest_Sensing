# CS-598--Forest_Sensing

Steps to Tree Segementation Code:
1. Python 3.6 version recommended
2. Install `requirements.txt` by `pip install -r requirements.txt`. 
3. Navigate to the `tree segemenation folder`.
4. Run `train.py`
5. To visualize the segmentation from the pretrained model at an inference stage , run the `inference_script.ipynb`.

Steps to run the tree species classification:
1. Python 3.6 version recommended
2. Install `requirements.txt` by `pip install -r requirements.txt`.
3. Navigate to the `tree species classifcation folder`.
4. Unzip the files in `data` folder and save them in the same name `train` and `test`. 
5. Ensure that `train` and `test` are the folder names.
6. Run the `.ipynb` file cell by cell to see the results

Steps to run planning:
1. Python libraries os, numpy, scipy, pandas, seaborn, matplotlib, and pickle required. Version must be recent. Note: seaborn and pandas must be compabtible with each other. 
2. Navigate to `planning/src`.
3. Run `main.py` to run experiments. This will create `planning/outputs` and will fill with .png maps and .pkl data files.
4. Run `process_results.py` to process the results. This saves files `planning/outputs/all_data_${i}.pkl` with processed data in the form of numpy arrays. 
5. To unzip and process data, run `make_plots.py`. This file reads outputs from step 4, and creates one large dataframe. Results can be sorted, sifted, etc as desired by the user. The output plots for the current file are presented in our paper. 

`planning/inputs` contains the Google Maps inputs for the planning part of the project, as well as the 200ft-to-pixels ratio. 


Addtionally, we also tried classifying tree species using the canopy images from `Sierra Nevada forest`. This is a work in progress and still needs further investigation which is currently not under the present scopr of the project. 

1. Python 3.6 version recommended
2. Install `requirements.txt` by `pip install -r requirements.txt`.
3. Navigate to the `tree `Tree_Classification_Initial_Results` and run the `Training and Testing.ipynb`.

Steps to collect data with the Sensor Logic Inc's uwb radar:
1. In a windows machine, connect the radar.
2. Identify the usb port (go to Device Manager and check under USB Connector Managers)
3. Change the port in line 5 of `collect_data.m`
4. Run `collect_data.m`
5. Run `analyze_data.m`
