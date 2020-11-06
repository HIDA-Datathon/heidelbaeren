# heidelbaeren

In the course of the HIDA DATATHON for Grand Challenges on Climate Change 2020.   

### The challenge 
Using high level open source frameworks for
landscape image segmentation with PyTorch

### Our approach

The configuration

architecture | pyramid attention network (PAN);
encoder | se_resnext50_32x4d;
batchsize | 8;
learning rate | 0.0001;
activation | sigmoid;
data_size | 256 x 256

Used resources:

PyTorch: https://pytorch.org/

PyTorch Lightning: https://pytorchlightning.ai/

Segmentation models: https://github.com/qubvel/segmentation_models.pytorch

### Our results
Validation segmentation DICE score: ~ 95%
