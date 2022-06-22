# Thyroid Nodule Segmentation in Ultrasound Scans

This technique is divided into 2 stages. The first stage trains the network using a normal cross entropy loss and dice loss, this is done in file `stage1_training.py`.
Stage 2 then uses the weights from the stage 1 and pushes it's performance using Focal loss, this is done in fine `train_s.py`.
