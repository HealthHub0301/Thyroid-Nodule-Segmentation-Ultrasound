## Thyroid-Quantification-in-Ultrasound-Scan-

# Requirements
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
git clone https://github.com/HealthHub0301/Thyroid-Quantification-in-Ultrasound-Scan-.git
cd code
```
2. Download the DDTI data (http://cimalab.intec.co/?lang=en&mod=project&id=31) and put the data in `../data/DDTI or `../data/HH

3. Train the model
```
cd code
python train_2D.py 
```

4. Test the model
```
python test_2D_fully.py
```
# Method

* [Uncertainty Rectified Pyramid Consistency](https://arxiv.org/pdf/2012.07042.pdf)
