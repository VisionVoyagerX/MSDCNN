# MSDCNN

PNN pansharpenning method pytorch implementation

Based on implementation: https://github.com/xyc19970716/Deep-Learning-PanSharpening/tree/main

Paper link: https://www.mdpi.com/2072-4292/8/7/594](https://arxiv.org/pdf/1712.09809.pdf

# Torch Summary

```
==========================================================================================
Layer (type:depth-idx)                   Output Shape              Param #
==========================================================================================
├─Conv2d: 1-1                            [-1, 64, 256, 256]        25,984
├─ReLU: 1-2                              [-1, 64, 256, 256]        --
├─Conv2d: 1-3                            [-1, 32, 256, 256]        51,232
├─ReLU: 1-4                              [-1, 32, 256, 256]        --
├─Conv2d: 1-5                            [-1, 4, 256, 256]         3,204
==========================================================================================
Total params: 80,420
Trainable params: 80,420
Non-trainable params: 0
Total mult-adds (G): 5.26
==========================================================================================
Input size (MB): 0.25
Forward/backward pass size (MB): 50.00
Params size (MB): 0.31
Estimated Total Size (MB): 50.56
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
