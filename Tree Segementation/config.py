# Config file for DeepForest module

#cpu workers for data loaders
#Dataloaders
workers: 1
gpus: 
distributed_backend:
batch_size: 1

#Non-max supression of overlapping predictions
nms_thresh: 0.05
score_thresh: 0.1

train:

    csv_file:
    root_dir:
    
    #Optomizer  initial learning rate
    lr: 0.001

    #Print loss every n epochs
    epochs: 1
    #Useful debugging flag in pytorch lightning, set to True to get a single batch of training to test settings.
    fast_dev_run: False
    
validation:
    #callback args
    csv_file: 
    root_dir:
    #Intersection over union evaluation
    iou_threshold: 0.4
    val_accuracy_interval: 5

optimizer=optim.SGD(self.model.parameters(), lr=self.config["train"]["lr"], momentum=0.9)

self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', 
                                                   factor=0.1, patience=10, 
                                                   verbose=True, threshold=0.0001, 
                                                   threshold_mode='rel', cooldown=0, 
                                                   min_lr=0, eps=1e-08)