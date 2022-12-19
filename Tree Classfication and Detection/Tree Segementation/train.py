#load the modules
import os
import time
import numpy as np
from deepforest import main 
from deepforest import get_data
from deepforest import utilities
from deepforest import preprocess

#convert hand annotations from xml into retinanet format
#The get_data function is only needed when fetching sample package data

YELL_xml = get_data("2019_YELL_2_528000_4978000_image_crop2.xml")
annotation = utilities.xml_to_annotations(YELL_xml)
annotation.head()

#load the image file corresponding to the annotaion file
YELL_train = get_data("2019_YELL_2_528000_4978000_image_crop2.png")
image_path = os.path.dirname(YELL_train)
#Write converted dataframe to file. Saved alongside the images
annotation.to_csv(os.path.join(image_path,"train_example.csv"), index=False)

"""

->Prepare Training and Validation Data
i)75% Training Data
ii) 25% Validation Data

"""

#Find annotation path
annotation_path = os.path.join(image_path,"train_example.csv")
#crop images will save in a newly created directory
#os.mkdir(os.getcwd(),'train_data_folder')
crop_dir = os.path.join(os.getcwd(),'train_data_folder')
train_annotations= preprocess.split_raster(path_to_raster=YELL_train,
                                 annotations_file=annotation_path,
                                 base_dir=crop_dir,
                                 patch_size=400,
                                 patch_overlap=0.05)


#Split image crops into training and test. Normally these would be different tiles! Just as an example.
image_paths = train_annotations.image_path.unique()
#split 25% validation annotation
valid_paths = np.random.choice(image_paths, int(len(image_paths)*0.25) )
valid_annotations = train_annotations.loc[train_annotations.image_path.isin(valid_paths)]
train_annotations = train_annotations.loc[~train_annotations.image_path.isin(valid_paths)]


#View output
train_annotations.head()
print("There are {} training crown annotations".format(train_annotations.shape[0]))
print("There are {} test crown annotations".format(valid_annotations.shape[0]))

#save to file and create the file dir
annotations_file= os.path.join(crop_dir,"train.csv")
validation_file= os.path.join(crop_dir,"valid.csv")
#Write window annotations file without a header row, same location as the "base_dir" above.
train_annotations.to_csv(annotations_file,index=False)
valid_annotations.to_csv(validation_file,index=False)


#initial the model and change the corresponding config file
m = main.deepforest()
m.config["train"]["csv_file"] = annotations_file
m.config["train"]["root_dir"] = os.path.dirname(annotations_file)
#Since this is a demo example and we aren't training for long, only show the higher quality boxes
m.config["score_thresh"] = 0.4
m.config["train"]['epochs'] = 2
m.config["validation"]["csv_file"] = validation_file
m.config["validation"]["root_dir"] = os.path.dirname(validation_file)
#create a pytorch lighting trainer used to training 
m.create_trainer()
#load the lastest release model 
m.use_release()

start_time = time.time()
m.trainer.fit(m)
print(f"--- Training on CPU: {(time.time() - start_time):.2f} seconds ---")

#create a directory to save the predict image
save_dir = os.path.join(os.getcwd(),'pred_result')
try:
  os.mkdir(save_dir)
except FileExistsError:
  pass
results = m.evaluate(annotations_file, os.path.dirname(annotations_file), iou_threshold = 0.4, savedir = save_dir)

print("Results: ",results)
print("Box_Predictions:",results['box_precision'])
print("Box_Recall: ",results["box_recall"])
print("Results_Main: ", results["results"])
print("Class_Recall: ",results["class_recall"])


