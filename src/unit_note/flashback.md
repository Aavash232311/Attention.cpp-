The Standard Softmax logic from the flash attention paper

S = QKT, P = softmax(S), O = PV

Let's take O = PV

In topics like thermodynamics ( for different thermodynamics processes ) or let's say wave function we do partial differentiation but here dO is the output of matrix multiplication between P and value “v”.

dO = P dV + v dP 

In neural networks we care about L(0) because it tells us how to change the model's parameters to reduce the loss.

Let us consider a flow state

Input -> Layer A -> Layer B -> Loss L

When we are trying to compute the gradient for Layer A, the gradient coming from the B is called the upstream gradient.

For a scaler function L(0)

Note:- : meaning inner product basically multiply each element and then sum.

dL = i jLOij dOij

dL = LO : dO


The upstream gradient G = LO

dL = G : (P dV + V dP)
dL = G : (P dV) + G : (dPV)

Lets write the above term in index notation again:

G : (P dV) = i jGij (P dV)ij  - i)
G : (V dP) = i j Gij (V dP)ij - ii)

Lets understand why that transpose came into place by proving one of the equations.

From equation i)

G : (P dV) = i j Gij (P dV)ij

Let us only take the term P dV ( Note:- this is not thermodynamics or waive function the output is defined as the result of something so we cannot keep something constant) 

(P dV)ij = KPik(dV)kj - iii) 

Note:- From the attention score formula P which is the output of the softmax and then V which is the value is matrix multiplied. 

Now multiply that term (P dV)ij with upstream gradient G

From equation iii)

Pik (dV)kj and Gij will have an inner product. 

G : (P dV) = i j kGij Pik (dV)kj

We are now regrouping by multiplying with (dV)kj

k j( iPik Gij ) (dV)kj -iv) Basically you can group this with the involved terms.

Now by the definition of transpose from school level math.

(PT)ki = Pik common point of confusing here value of the index are same

ki(iPTki Gij ) (dV)kj  from eq iv)

G: (P dV ) = (PT G) : dV



Reading off the gradient for the first and second part of the above equation.

dL = (PTG) : dV              - 

LV = PT LO - a)

dL = (GVT) : dP               - 

LP = LO VT - b)