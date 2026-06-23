### Forward pass logic

$$
y^{*} = \text{softmax}(z)
$$

$$
L = - \sum_i y_i \log y^{*}_i
$$

$$
\delta_{\text{logits}} = \frac{\partial L}{\partial z} = y^{*} - y
$$

---

### Expanding the loss

$$
L = - \sum_i y_i \log y^{*}_i
$$

Now take partial derivative w.r.t. $z_k$:

$$
\frac{\partial}{\partial z_k}(y_i \log y^{*}_i) = y_i \frac{\partial}{\partial z_k}(\log y^{*}_i) \tag{i}
$$

Recalling the chain rule of derivatives:

$$
\frac{d}{dx}(\log u) = \frac{d \log u}{du} \cdot \frac{du}{dx}
$$

Plugging this back into equation i:

$$
\frac{\partial (y_i \log y^{*}_i)}{\partial z_k} = y_i \cdot \frac{1}{y^{*}_i} \cdot \frac{\partial y^{*}_i}{\partial z_k}
$$

$$
\frac{\partial (y_i \log y^{*}_i)}{\partial z_k} = y_i \cdot \frac{1}{y^{*}_i} \cdot y^{*}_i (\delta_{ik} - y^{*}_k)
$$

$$
\frac{\partial (y_i \log y^{*}_i)}{\partial z_k} = y_i (\delta_{ik} - y^{*}_k)
$$

$$
\frac{\partial (y_i \log y^{*}_i)}{\partial z_k} = y_i \delta_{ik} - y_i \ y^{*}_k
$$

Now the function


$$
\frac{\partial L}{\partial z_k} = -\sum_i  (y_i \delta_{ik} - y_i \ y^{*}_k)
$$

Now let's split the sum:

$$-\sum_i y_i \delta_{ik} + \sum_i y_i y^*_k$$

Since $\delta_{ik} = 1$ if and only if $i = k$ (and $0$ otherwise), the first summation collapses to a single term where the index $i$ is replaced by $k$:

$$\sum_i y_i \delta_{ik} = y_k$$

<i>y is a one-hot vector so all its values sum to 1</i>

Now lets take a look at the second term

$$\sum_i y_i y^*_k$$

$$y^*_k \sum_i y_i$$

Now because of one hot encode giving us yk:-
$$\sum_i y_i = 1$$

$$-y_k + y^*_k = y^*_k - y_k = \frac{\partial L}{\partial z_k}$$



<hr />

Lets recall the derivative of softmax

Let the row-wise denominator sums be defined as:
* Row 1: $\sum e^{s_1} = e^{s_{11}} + e^{s_{12}} + e^{s_{13}}$
* Row 2: $\sum e^{s_2} = e^{s_{21}} + e^{s_{22}} + e^{s_{23}}$
* Row 3: $\sum e^{s_3} = e^{s_{31}} + e^{s_{32}} + e^{s_{33}}$

$$P = \text{softmax}(S) = \begin{bmatrix}
\frac{e^{s_{11}}}{\sum e^{s_1}} & \frac{e^{s_{12}}}{\sum e^{s_1}} & \frac{e^{s_{13}}}{\sum e^{s_1}} \\
\frac{e^{s_{21}}}{\sum e^{s_2}} & \frac{e^{s_{22}}}{\sum e^{s_2}} & \frac{e^{s_{23}}}{\sum e^{s_2}} \\
\frac{e^{s_{31}}}{\sum e^{s_3}} & \frac{e^{s_{32}}}{\sum e^{s_3}} & \frac{e^{s_{33}}}{\sum e^{s_3}}
\end{bmatrix} = \begin{bmatrix} 
p_{11} & p_{12} & p_{13} \\ 
p_{21} & p_{22} & p_{23} \\ 
p_{31} & p_{32} & p_{33} 
\end{bmatrix}$$



$$
J(P_1)
=
\begin{bmatrix}
p_{11}(1-p_{11}) & -p_{11}p_{12} & -p_{11}p_{13} \\
-p_{11}p_{12} & p_{12}(1-p_{12}) & -p_{12}p_{13} \\
-p_{11}p_{13} & -p_{12}p_{13} & p_{13}(1-p_{13})
\end{bmatrix}
$$

Case 1: $i = k$

$$f(z_k) = y^*_i(1 - y^*_i)$$

Case 2: $i \neq k$

$$f(z_k) = -y^*_i y^*_k$$

---

$$\delta = y^* - y$$

derived in flashback.md

$$
J(y^{*}) = \text{diag}(y^{*}) - y^{*} {y^{*}}^T
$$

$$
\frac{\partial y^{*}_i}{\partial z_k} = J_{ik}
$$





### Now lets take a look at lm head

This is a linear transformation task. 
<i><b>
Many years ago I might have used this in the output layer of ANN when doing backpropagation from scratch in java.
</b></i>

Like in the derivation of FlashAttention, we will look at the matrix version of the chain rule to get a better idea of why the transposes appear.

$$Y = AX$$

$$\frac{\partial L}{\partial X} = A^T \frac{\partial L}{\partial Y}$$

$$\frac{\partial L}{\partial A} = \frac{\partial L}{\partial Y} X^T$$

<hr />

$$z = Wh + b$$

Now gradient with respect to W:

$$\frac{\partial z_i}{\partial W_i} = h$$

$$\frac{\partial L}{\partial W_i} = \frac{\partial L}{\partial z_i} \cdot \frac{\partial z_i}{\partial W_i}$$

From the above equations:

$$\frac{\partial L}{\partial W_i} = (y^* - y)h$$

Stacking all the rows:

$$\frac{\partial L}{\partial W} = (y^* - y) h^T$$

Now, the gradient with respect to $b$:

$$\frac{\partial z_i}{\partial b_i} = 0 + \frac{\partial b_i}{\partial b_i}$$

$$\frac{\partial z_i}{\partial b_i} = 1$$

$$\frac{\partial L}{\partial b} = y^* - y$$

Similarly, the gradient with respect to $h$:

$$z = Wh + b$$

$$\frac{\partial z}{\partial h} = W$$

$$\frac{\partial L}{\partial h} = \frac{\partial L}{\partial z} \frac{\partial z}{\partial h} 
$$

$$\frac{\partial L}{\partial h} = W^T \frac{\partial L}{\partial z}$$


Finally,

$$\delta = y^* - y$$

$$\frac{\partial L}{\partial z} = \delta$$

For weights:

$$\frac{\partial L}{\partial W} = \delta h^T$$

For bias:

$$\frac{\partial L}{\partial b} = \delta$$

For $h$:

$$\frac{\partial L}{\partial h} = W^T \delta$$


Now `output_proj` inside of the attention head is also a linear transformation.

$$z = Wx + b$$

Upstream gradient:

$$\delta = \frac{\partial L}{\partial z}$$

$$\frac{\partial L}{\partial W} = \delta x^T$$

$$\frac{\partial L}{\partial x} = W^T \delta$$

$$\frac{\partial L}{\partial b} = \delta$$

The linear layer does not generate the upstream gradient; rather, it gets passed along. For example, for $O = PV$, the gradient originates from the Cross-Entropy loss and the softmax function.