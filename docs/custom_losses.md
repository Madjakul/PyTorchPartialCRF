# List of Changes

Small adjustements have been made to the original code to suit my need. This code is not a module anymore, but rather a script to add to a project. The ``CRF`` and ``MarginalCRF`` have been removed to avoid redundancies. Some useless variables have been removed and some tensors/parameters' declaration changed. Buth the main goal of this was to be able to use som custom loss functions, not only the negative log-likelihood that tends to overfit noise.



## Custom Losses

### Negative Log-Likelihood

Default loss function used to optimize the CRF:

```{math}
\mathcal{L}_{NLL} = -log\left[\frac{exp(\sum_{i = 1}^{I}{E(\textbf{x}_{i}, y_{i})} + \sum_{i = 1}^{I}{T(y_{i}, y_{i - 1})})}{Z(\textbf{X})} \right]
```


### Corrected Negative Log-Likelihood

Corrected negative log-likelihood [1] using negative log-unlikelihood [2] as a regularizer.

```{math}
\mathcal{L}_{CNLL} = P(\textbf{y}_{pred} = \textbf{y}_{true} | E(\textbf{X}, \textbf{y}_{pred}))\mathcal{L}_{NLL} + P(\textbf{y}_{pred} \neq \textbf{y}_{true} | E(\textbf{X}, \textbf{y}_{pred}))\mathcal{L}_{NLU}
```


### Generelaized Cross-Entropy

Noise-robust cross-entropy loss function [2] [3] using the marginal probabilities of each tokens. ``q`` is a hyperparameter.

```{math}
\mathcal{L}_{GCE} = \sum_{i = 0}^{I}{\frac{1 - P(y_{i}, \textbf{x}_{i})^{q}}{q}}
```


## References

[1] Welleck, Sean, et al. “Neural text generation with unlikelihood training.” arXiv preprint arXiv:1908.04319 (2019).

[2] Jiang, Haoming, et al. “Named entity recognition with small strongly labeled and large weakly labeled data.” arXiv preprint arXiv:2106.08977 (2021).

[3] Zhang, Zhilu, and Mert Sabuncu. “Generalized cross entropy loss for training deep neural networks with noisy labels.” Advances in neural information processing systems 31 (2018).

[4] Meng, Yu, et al. “Distantly-supervised named entity recognition with noise-robust learning and language model augmented self-training.” arXiv preprint arXiv:2109.05003 (2021).