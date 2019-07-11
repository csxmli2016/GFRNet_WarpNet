 Pytorch implementation <B>WarpNet</B> for [Learning Warped Guidance for Blind Face Restoration](https://arxiv.org/abs/1804.04829)
 
 # WarpNet framework
This is only the subnet of GFRNet. The <B>WarpNet</B> takes the degraded observation and guided image as input to predict the dense flow field, which is adopted to deform guided image to the warped guidance. Network architecture is shown below.

<img src="./imgs/warpnet.png">
