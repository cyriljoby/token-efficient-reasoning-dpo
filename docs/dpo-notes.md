## DPO: 
- Instead of training a reward model and running RL against it, DPO skips straight to a loss function that increases probability of preferred responses and the opposite for rejected responses
- ```loss = -log(σ(β · [log(π_θ(y_w|x)/π_ref(y_w|x)) − log(π_θ(y_l|x)/π_ref(y_l|x))]))```
    - y_w: winning response
    - y_l: losing/rejected response
    - π_θ: model currently being trained
    - π_ref: frozen copy of the model before training
    - σ: sigmoid, which turns diff between log ratioes into a probability
    - β: temperature-like hyperparameter controlling how aggressive the optimization is
        - as β increase, the the lower the deviation from π_ref(the reference model)
    - For each pair of responses, this is how the loss is calculated:
        - compute the ratio of new to old for both the winning and losing reponses
        - calculate the difference between these log ratioes
        - feed into a sigmoid: this is the **preference probability**, which represents how confident we are that the new model favors y_w over y_l
        - feed the preference probability into -log to convert it into a loss function. this way high probabilities become low penalties and low probabilities become high penalites
    - After finding the loss:
        - Backpropagate: compute gradient using backpropagation on the trainable model(π_θ). now we know which direction would make loss the smallest
        - Update weights: update every weight in the opposite direction of the gradient, using a learning rate to ensure convergence
        - Repeat for every prompt, chosen, reject triple

---

## What π(y|x) actually is (the detail that matters most here)

`log π(y|x)` is not one number the model hands you — it is the **sum of per-token log-probs over the completion tokens(thinking tokens + answer token) only**:

```
log π(y|x) = Σ_t log π(y_t | x, y_<t)
```
- the probability of the whole response is built up by computing the conditional probability of each token


Implementation consequences:
- Mask out the prompt tokens and the padding. Only completion tokens count.
- **Sum, not mean.** Vanilla DPO sums. Taking the mean instead gives you a *length-normalized* objective — a different method (that is essentially what SimPO does), not an implementation detail.

## DPO has no seperate/trained reward model, it is implicit

DPO's derivation starts from the KL-constrained RLHF objective. Its optimal policy is

```
π*(y|x) = (1/Z(x)) · π_ref(y|x) · exp(r(x,y)/β)
```
Start at the reference model's distribution → boost or penalize each response based on its reward → renormalize with Z(x) so it's a valid probability distribution again.

Solve for r:
```
r(x,y) = β log(π*(y|x)/π_ref(y|x)) + β log Z(x)
```
Z is impossible to compute, since it the sum over every possible response in the entire space of possible text outputs. But, we dont need to know the absolute reward, we are interested in the difference between the winning and losing rewards, also known as the Bradley-Terry model.

```
P(y_w ≻ y_l) = σ(r(x,y_w) − r(x,y_l))
```

Substituing our reward function in:
```
r(x,y_w) − r(x,y_l) = [β log(π*(y_w|x)/π_ref(y_w|x)) + β log Z(x)] − [β log(π*(y_l|x)/π_ref(y_l|x)) + β log Z(x)]
```
Cancelling out the the β log Z(x) terms
```
r(x,y_w) − r(x,y_l) = β log(π*(y_w|x)/π_ref(y_w|x)) − β log(π*(y_l|x)/π_ref(y_l|x))
```
And then plugging back into Bradley-Terry gives us our "implicit" reward in the form of the loss function:
```
loss = -log σ(β log(π_θ(y_w|x)/π_ref(y_w|x)) − β log(π_θ(y_l|x)/π_ref(y_l|x)))
```

## Gradient shape

```
∇L = −β · E[ σ(r̂_l − r̂_w) · (∇log π_θ(y_w|x) − ∇log π_θ(y_l|x)) ]
```

The `σ(r̂_l − r̂_w)` factor is an automatic difficulty weight: pairs the model already gets right contribute a small gradient, pairs it gets backwards contribute a large one.