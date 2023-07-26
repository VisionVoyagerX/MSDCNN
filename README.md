# MSDCNN

PNN pansharpenning method pytorch implementation

Based on implementation: https://github.com/xyc19970716/Deep-Learning-PanSharpening/tree/main


Paper link: https://ieeexplore.ieee.org/abstract/document/8127731

# Torch Summary

```
==========================================================================================
Layer (type:depth-idx)                   Output Shape              Param #
==========================================================================================
MSDCNN_model                             [1, 8, 256, 256]          --
├─Conv2d: 1-1                            [1, 64, 256, 256]         46,720
├─ReLU: 1-2                              [1, 64, 256, 256]         --
├─Conv2d: 1-3                            [1, 32, 256, 256]         2,080
├─ReLU: 1-4                              [1, 32, 256, 256]         --
├─Conv2d: 1-5                            [1, 8, 256, 256]          6,408
├─Conv2d: 1-6                            [1, 60, 256, 256]         26,520
├─ReLU: 1-7                              [1, 60, 256, 256]         --
├─Conv2d: 1-8                            [1, 20, 256, 256]         10,820
├─ReLU: 1-9                              [1, 20, 256, 256]         --
├─Conv2d: 1-10                           [1, 20, 256, 256]         30,020
├─ReLU: 1-11                             [1, 20, 256, 256]         --
├─Conv2d: 1-12                           [1, 20, 256, 256]         58,820
├─ReLU: 1-13                             [1, 20, 256, 256]         --
├─Conv2d: 1-14                           [1, 30, 256, 256]         16,230
├─ReLU: 1-15                             [1, 30, 256, 256]         --
├─Conv2d: 1-16                           [1, 10, 256, 256]         2,710
├─ReLU: 1-17                             [1, 10, 256, 256]         --
├─Conv2d: 1-18                           [1, 10, 256, 256]         7,510
├─ReLU: 1-19                             [1, 10, 256, 256]         --
├─Conv2d: 1-20                           [1, 10, 256, 256]         14,710
├─ReLU: 1-21                             [1, 10, 256, 256]         --
├─Conv2d: 1-22                           [1, 8, 256, 256]          6,008
==========================================================================================
Total params: 228,556
Trainable params: 228,556
Non-trainable params: 0
Total mult-adds (G): 14.98
==========================================================================================
Input size (MB): 0.39
Forward/backward pass size (MB): 153.09
Params size (MB): 0.91
Estimated Total Size (MB): 154.40
==========================================================================================
```

# Quantitative Results

## GaoFen-2

![alt text](https://github.com/nickdndndn/MSDCNN/blob/main/results/Figures_GF2.png?raw=true)

## WorldView-3

![alt text](https://github.com/nickdndndn/MSDCNN/blob/main/results/Figures_WV3.png?raw=true)

# Qualitative Results

## GaoFen-2

![alt text](https://github.com/nickdndndn/MSDCNN/blob/main/results/Images_GF2.png?raw=true)

## WorldView-3

![alt text](https://github.com/nickdndndn/MSDCNN/blob/main/results/Images_WV3.png?raw=true)
