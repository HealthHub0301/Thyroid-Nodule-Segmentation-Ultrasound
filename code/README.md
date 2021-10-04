

## Requirements
Some important required packages include:
* [Pytorch][torch_link] version >=0.4.1.
* TensorBoardX
* Python == 3.6 
* Efficientnet-Pytorch `pip install efficientnet_pytorch`
* Some basic python packages such as Numpy, Scikit-image, SimpleITK, Scipy ......

Follow official guidance to install [Pytorch][torch_link].

[torch_link]:https://pytorch.org/

# Usage

1. Clone the repo:
```
git clone https://https://github.com/HiLab-git/SSL4MIS.git 
cd SSL4MIS
```
2. Download the processed data and put the data in `../data/BraTS2019` or `../data/ACDC`, please read and follow the [README](https://github.com/Luoxd1996/SSL4MIS/tree/master/data/).

3. Train the model
```
cd code
python train_XXXXX_3D.py or python train_XXXXX_2D.py or bash train_acdc_XXXXX.sh
```

4. Test the model
```
python test_XXXXX.py
```
# Method

* [Uncertainty Rectified Pyramid Consistency](https://arxiv.org/pdf/2012.07042.pdf)
