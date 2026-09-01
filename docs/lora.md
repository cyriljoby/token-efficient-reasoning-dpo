## Lora: Low-rank adaptation
- A neural network layer has a weight matrix, W. LoRA says, don't change W. Instead create an update matrix, ΔW.
    - new weights = W + ΔW
- But, training ΔW would be the same size as W. So not much saved. Instead represent it as a product of two matrices.
    - ΔW = BA
- Matrices B and A are of the format (rows x r )and  (r x cols), where r is the maximum rank of matrix ΔW.
    - When fine-tuning a pretrained LLM with LoRA, instead of updating all the original parameters, we freeze them and train a much smaller set of new parameters (A, B).
- Forward Pass:
    ```
    y = Wx
    y' = (W + ΔW)x 
    y' = (W + BA)x 
    y' = Wx+BAx
    ```
- During backpropagation W is frozen while A, B are trainable
    - As A, B change so does ΔW. 
- At the start of LoRA training A is initialized with small random values and B to zeroes.
- During LoRA the updates are scaled using the LoRA scaling factor to control strength of lora update:
$y' = Wx + \frac{\alpha}{r}BAx$
- LoRA adapters are added to specific weight matrices, called LoRA target modules.

# Connection to DPO
During each step of DPO, a loss is calculated. The loss is used to calculate the gradient. But, instead of applying gradient descent directly to W, it updates A, B which changes ΔW. For each LoRA-adapted module in the policy:

$W_{\text{policy}} = W + \frac{\alpha}{r}\Delta W = W + \frac{\alpha}{r}BA$

**Impact**: Instead of having to maintain two models, π_ref and π_theta, we maintain W which is the large, pretrained weights along with A, B which are much smaller LoRA matrices

