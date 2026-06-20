Now that we have the backpropagation for attention head let us understand how does gradient flows between things like softmax + corss entropy loss, lm head, and output project.


Here is the forward pass logic:

$$y^* = \text{softmax}(z)$$

$$L = - \sum (y \log y^*)$$

$$\delta_{\text{logits}} = \frac{\partial L}{\partial z} = y^* - y$$



