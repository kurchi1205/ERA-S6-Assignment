# ERA-S6-Assignment

## Part 1
Following the screenshot of the excel sheet:

<img width="1069" alt="Screen Shot 2023-06-10 at 12 13 34 AM" src="https://github.com/kurchi1205/ERA-S6-Assignment/assets/40196782/c5fe9eb4-0089-4f0b-a65c-06ed973c5f1e">

### Major Steps in Backpropagation

- For BackPropagation to happen, first forward propagation must happen
- h1 is first calculated based as `w1*i1 + w2*i2`
- Activation function is applied on h1 to get a_h1.
- Similarly for h2 the calculation is `w3*i1 + w4*i2`.
- Activation function is applied on h2 to get a_h2.
- Output neurons are calculated similarly with a_h1 and a_h2 as input neurons and w5, w6, w7 and w8 as weights
- Activation function is applied on the output neurons o1 and o2 to get a_o1 and a_o2.
- Individual loss (E1 and E2) are calculated using predefined formulae. Then E_Total is updated.

Sole objective of BackPropagation is to update the weights so that desired loss is achieved.

- For that we need to calculate, how oes E_total change for slight change in the weights.
- We need to go back step by step . First how E1 and E2 are affected by these weights.
- E1 and E2 change with respect to a_o1 and a_o2 is calculated.
- a_o1 and a_o2 change with respect to o1 and o2 are calculated.
- o1 and o2 change with respect to w5, w6, w7 and w8 are calculated.
- E1 and E2 change with respect to a_h1 and a_h2 is calculated.
- a_h1 and a_h2 change with respect to h1 and h2 are calculated.
- h1 and h2 change with respect to w1, w2, w3 and w4 are calculated.

Thus in backpropagation, we follow the same path for forward propagation except in backward direction.

- Weights are updated as `wn' = wn - learning_rate * delta change in loss with respect to wn`
