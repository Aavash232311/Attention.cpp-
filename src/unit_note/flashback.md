The Standard Softmax logic from the flash attention paper

$$S = QK^T, \quad P = \text{softmax}(S), \quad O = PV$$

Let's take $O = PV$

In topics like thermodynamics ( for different thermodynamics processes ) or let's say wave function we do partial differentiation but here $dO$ is the output of matrix multiplication between $P$ and value $V$.

$$dO = P \, dV + dP \, V$$

In neural networks we care about $L(O)$ because it tells us how to change the model's parameters to reduce the loss.

Let us consider a flow state:

$$\text{Input} \rightarrow \text{Layer A} \rightarrow \text{Layer B} \rightarrow \text{Loss } L$$

When we are trying to compute the gradient for Layer A, the gradient coming from B is called the upstream gradient.

For a scalar function $L(O)$

> **Note:** $:$ meaning inner product basically multiply each element and then sum.

$$dL = \sum_{i,j} (L_O)_{ij} \, dO_{ij}$$

$$dL = L_O : dO$$

The upstream gradient $G = L_O$

$$dL = G : (P \, dV + dP \, V)$$
$$dL = G : (P \, dV) + G : (dP \, V)$$

Lets write the above term in index notation again:

$$G : (P \, dV) = \sum_{i,j} G_{ij} (P \, dV)_{ij} \quad \text{--- i)}$$
$$G : (dP \, V) = \sum_{i,j} G_{ij} (dP \, V)_{ij} \quad \text{--- ii)}$$

Lets understand why that transpose came into place by proving one of the equations.

From equation i)

$$G : (P \, dV) = \sum_{i,j} G_{ij} (P \, dV)_{ij}$$

Let us only take the term $P \, dV$ ( Note:- this is not thermodynamics or waive function the output is defined as the result of something so we cannot keep something constant) 

$$(P \, dV)_{ij} = \sum_{k} P_{ik} (dV)_{kj} \quad \text{--- iii)}$$

> **Note:** From the attention score formula $P$ which is the output of the softmax and then $V$ which is the value is matrix multiplied. 

Now multiply that term $(P \, dV)_{ij}$ with upstream gradient $G$

From equation iii)

$P_{ik} (dV)_{kj}$ and $G_{ij}$ will have an inner product. 

$$G : (P \, dV) = \sum_{i,j,k} G_{ij} P_{ik} (dV)_{kj}$$

We are now regrouping by multiplying with $(dV)_{kj}$

$$\sum_{k,j} \left( \sum_{i} P_{ik} G_{ij} \right) (dV)_{kj} \quad \text{--- iv)}$$

Basically you can group this with the involved terms.

Now by the definition of transpose from school level math.

$$(P^T)_{ki} = P_{ik}$$

Common point of confusing here value of the index are same:

$$\sum_{k,j} \left( \sum_{i} (P^T)_{ki} G_{ij} \right) (dV)_{kj} \quad \text{from eq iv)}$$

$$G : (P \, dV) = (P^T G) : dV$$

Reading off the gradient for the first and second part of the above equation.

$$dL = (P^T G) : dV$$

$$L_V = P^T L_O \quad \text{--- a)}$$

$$dL = (G V^T) : dP$$

$$L_P = L_O V^T \quad \text{--- b)}$$

Let's unfold exactly what is going on at the matrix level.

### 1. Value Matrix ($V$)
The Value matrix has shape $(3 \times 3)$:
$$V = \begin{bmatrix} 
v_{11} & v_{12} & v_{13} \\ 
v_{21} & v_{22} & v_{23} \\ 
v_{31} & v_{32} & v_{33} 
\end{bmatrix}$$

---

### 2. Attention Probability Matrix ($P$)
Each row of $P$ is the result of applying the `softmax` function across the corresponding row of the attention score matrix $S$. 

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

---

### 3. Output Matrix ($O = PV$)
Multiplying rows of $P$ by columns of $V$ gives us the fully expanded elements of the output matrix $O$:

$$O = \begin{bmatrix}
p_{11}v_{11} + p_{12}v_{21} + p_{13}v_{31} & p_{11}v_{12} + p_{12}v_{22} + p_{13}v_{32} & p_{11}v_{13} + p_{12}v_{23} + p_{13}v_{33} \\
p_{21}v_{11} + p_{22}v_{21} + p_{23}v_{31} & p_{21}v_{12} + p_{22}v_{22} + p_{23}v_{32} & p_{21}v_{13} + p_{22}v_{23} + p_{23}v_{33} \\
p_{31}v_{11} + p_{32}v_{21} + p_{33}v_{31} & p_{31}v_{12} + p_{32}v_{22} + p_{33}v_{32} & p_{31}v_{13} + p_{32}v_{23} + p_{33}v_{33}
\end{bmatrix}$$

Which maps directly to the standard layout:
$$O = \begin{bmatrix} 
O_{11} & O_{12} & O_{13} \\ 
O_{21} & O_{22} & O_{23} \\ 
O_{31} & O_{32} & O_{33} 
\end{bmatrix}$$