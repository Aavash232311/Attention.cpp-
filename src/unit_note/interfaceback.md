Now that we have the backpropagation for attention head let us understand how does gradient flows between things like softmax + corss entropy loss, lm head, and output project.



### Forward pass logic

$$
y^* = \text{softmax}(z)
$$

$$
L = - \sum_i (y_i \log y^*_i)
$$

$$
\delta_{\text{logits}} = \frac{\partial L}{\partial z} = y^* - y
$$

---

### Expanding the loss

$$
L = - \sum_i \left(y_i \log(\text{softmax}(y^*)_i)\right)
$$

Now take partial derivative w.r.t. \(z_k\):

$$
\frac{\partial}{\partial z_k}(y_i \log y^*_i)
= y_i \frac{\partial}{\partial z_k}(\log y^*_i)
$$  equation i)


Recalling the chain rule of derivatives:


$$\frac{d}{dx}(\log u) = \frac{d \log u}{du} \cdot \frac{du}{dx}$$


Plugging this back into equation i):

$$\frac{\partial (y_i \log y^*_i)}{\partial z_k} = y_i \cdot \frac{1}{p^i} \frac{\partial y^*_i}{\partial z_k}$$


derived in flashback.md

$$ 
J(P) = \text{diag}(P) - PP^T
$$

$$ 
\frac{\partial y^*_i}{\partial z_k} =Jik

$$

---




