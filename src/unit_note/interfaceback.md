### Forward pass logic

$$
\hat{y} = \text{softmax}(z)
$$

$$
L = - \sum_i y_i \log \hat{y}_i
$$

$$
\delta_{\text{logits}} = \frac{\partial L}{\partial z} = \hat{y} - y
$$


### Expanding the loss

$$
L = - \sum_i y_i \log \hat{y}_i
$$

Now take partial derivative w.r.t. $z_k$:

Equation (i):

$$
\frac{\partial}{\partial z_k}(y_i \log \hat{y}_i) = y_i \frac{\partial}{\partial z_k}(\log \hat{y}_i)
$$

Recalling the chain rule of derivatives:

$$
\frac{d}{dx}(\log u) = \frac{d \log u}{du} \cdot \frac{du}{dx}
$$

Plugging this back into equation (i):

$$
\frac{\partial (y_i \log \hat{y}_i)}{\partial z_k} = y_i \cdot \frac{1}{\hat{y}_i} \cdot \frac{\partial \hat{y}_i}{\partial z_k}
$$

$$
\frac{\partial (y_i \log \hat{y}_i)}{\partial z_k} = y_i \cdot \frac{1}{\hat{y}_i} \cdot \hat{y}_i (\delta_{ik} - \hat{y}_k)
$$

$$
\frac{\partial (y_i \log \hat{y}_i)}{\partial z_k} = y_i (\delta_{ik} - \hat{y}_k)
$$

$$
\frac{\partial (y_i \log \hat{y}_i)}{\partial z_k} = y_i \delta_{ik} - y_i \hat{y}_k
$$

Now the function

$$
\frac{\partial L}{\partial z_k} = -\sum_i  (y_i \delta_{ik} - y_i \hat{y}_k)
$$

Now let's split the sum:

$$
-\sum_i y_i \delta_{ik} + \sum_i y_i \hat{y}_k
$$

Since $\delta_{ik} = 1$ if and only if $i = k$ (and $0$ otherwise), the first summation collapses to a single term where the index $i$ is replaced by $k$:

$$
\sum_i y_i \delta_{ik} = y_k
$$

<i>y is a one-hot vector so all its values sum to 1</i>

Now lets take a look at the second term

$$
\sum_i y_i \hat{y}_k
$$

$$
\hat{y}_k \sum_i y_i
$$

Now because of one hot encode giving us $y_k$:

$$
\sum_i y_i = 1
$$

$$
-y_k + \hat{y}_k = \hat{y}_k - y_k = \frac{\partial L}{\partial z_k}
$$

---

derived in flashback.md

$$
J(\hat{y}) = \text{diag}(\hat{y}) - \hat{y} \hat{y}^T
$$

$$
\frac{\partial \hat{y}_i}{\partial z_k} = J_{ik}
$$

### Now lets take a look at lm head

This is a linear transformation task.

<i><b>
Many years ago I might have used this in the output layer of ANN when doing backpropagation from scratch in java.
</b></i>

Like in the derivation of FlashAttention, we will look at the matrix version of the chain rule to get a better idea of why the transposes appear.

$$
Y = AX
$$

$$
\frac{\partial L}{\partial X} = A^T \frac{\partial L}{\partial Y}
$$

$$
\frac{\partial L}{\partial A} = \frac{\partial L}{\partial Y} X^T
$$

---

$$
z = Wh + b
$$

Now gradient with respect to W:

$$
\frac{\partial z_i}{\partial W_i} = h
$$

$$
\frac{\partial L}{\partial W_i} = \frac{\partial L}{\partial z_i} \cdot \frac{\partial z_i}{\partial W_i}
$$

From the above equations:

$$
\frac{\partial L}{\partial W_i} = (\hat{y} - y)h
$$

Stacking all the rows:

$$
\frac{\partial L}{\partial W} = (\hat{y} - y) h^T
$$

Now, the gradient with respect to $b$:

$$
\frac{\partial z_i}{\partial b_i} = 0 + \frac{\partial b_i}{\partial b_i}
$$

$$
\frac{\partial z_i}{\partial b_i} = 1
$$

$$
\frac{\partial L}{\partial b} = \hat{y} - y
$$

Similarly, the gradient with respect to $h$:

$$
z = Wh + b
$$

$$
\frac{\partial z}{\partial h} = W
$$

$$
\frac{\partial L}{\partial h} = \frac{\partial L}{\partial z} \frac{\partial z}{\partial h}
$$

$$
\frac{\partial L}{\partial h} = W^T \frac{\partial L}{\partial z}
$$

Finally,

$$
\delta = \hat{y} - y
$$

$$
\frac{\partial L}{\partial z} = \delta
$$

For weights:

$$
\frac{\partial L}{\partial W} = \delta h^T
$$

For bias:

$$
\frac{\partial L}{\partial b} = \delta
$$

For $h$:

$$
\frac{\partial L}{\partial h} = W^T \delta
$$

Now `output_proj` inside of the attention head is also a linear transformation.

$$
z = Wx + b
$$

Upstream gradient:

$$
\delta = \frac{\partial L}{\partial z}
$$

$$
\frac{\partial L}{\partial W} = \delta x^T
$$

$$
\frac{\partial L}{\partial x} = W^T \delta
$$

$$
\frac{\partial L}{\partial b} = \delta
$$

The linear layer does not generate the upstream gradient; rather, it gets passed along. For example, for $O = PV$, the gradient originates from the Cross-Entropy loss and the softmax function.