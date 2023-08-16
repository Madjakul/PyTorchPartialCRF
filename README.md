# Pytorch Partial/Fuzzy CRF
---

This code is originally from [kajyuuen](https://github.com/kajyuuen).

This fork simply implements custom loss functions, alongside some minor optimizations to the code.



# Installation

## Dependencies

It only depends on ``torch``.

```sh
torch~=2.0.1
```


## Install

```sh
pip install -r requirements.txt
```



# Quickstart

Remember that $-1$ means unknown tag:

```py
>>> import torch
>>> from pytorch_partial_crf import PartialCRF
>>>
>>>
>>> batch_size = 2
>>> sequence_length = 9
>>> num_tags = 5
>>> emissions = torch.randn(batch_size, sequence_length, num_tags)
>>> emissions
tensor([[[-0.5437,  0.9088,  0.4173, -1.3075, -1.0963],
         [ 0.1396, -0.0843, -1.2068,  0.7572,  0.5796],
         [ 1.4185, -0.6221,  0.8547, -0.9173,  0.9208],
         [ 0.4390,  1.7294, -2.2982,  0.4782,  0.7222],
         [ 1.5666,  0.7675,  0.3230,  0.4046,  0.4232],
         [-0.4828,  0.8027, -0.0995,  1.4749,  0.4170],
         [-0.5631,  0.5672,  0.4975, -0.5789,  0.9422],
         [-0.0219,  0.1128,  0.9551,  0.0825, -0.8257],
         [ 0.2484,  0.1888,  0.6151, -0.7292, -1.6003]],

        [[ 0.4377, -0.2834, -0.0981, -0.5948, -1.9315],
         [-1.4660, -0.3846, -0.2995, -0.0706,  0.3094],
         [ 0.0249,  1.9489,  0.0665,  1.0557, -0.9480],
         [ 0.6224, -1.0894, -1.3665,  2.1289, -1.7502],
         [-0.7008, -0.5063,  0.6002, -1.3744,  0.0519],
         [ 1.4107, -0.9092,  1.7128, -0.9601, -1.0653],
         [ 0.6548,  0.8773, -0.4040,  0.2110,  1.2022],
         [ 0.0100,  0.9134, -0.2474,  0.2166, -0.1720],
         [ 0.3302,  2.0470,  0.2935,  0.3067,  0.0624]]])
>>>
>>> tags = torch.randint(0, 5, (batch_size, sequence_length))
>>> tags
tensor([[0, 0, 2, 2, 3, 2, 0, 3, 2],
        [1, 1, 4, 3, 0, 3, 4, 1, 1]])
>>>
>>> mask = torch.bernoulli(torch.empty(batch_size, sequence_length).uniform_(0, 1)).byte()
>>> mask
tensor([[1, 0, 0, 1, 0, 0, 1, 0, 1],
        [0, 1, 1, 1, 0, 0, 0, 1, 0]], dtype=torch.uint8)
```


## Using the Partial/Fuzzy CRF

### Decoding

```py
>>> crf = PartialCRF(num_tags, device="cpu")
>>> crf.viterbi_decode(emissions, mask)
[[1, 2, 1, 0], [1, 3, 1, 0]]
```

### Computing the Marginal Probabilities

```py
>>> crf.marginal_probabilities(emissions, mask)
tensor([[[0.0437, 0.4929, 0.3082, 0.0818, 0.0734],
         [0.2032, 0.2544, 0.2469, 0.2462, 0.0493]],

        [[0.0437, 0.4929, 0.3082, 0.0818, 0.0734],
         [0.0467, 0.1330, 0.2178, 0.3913, 0.2112]],

        [[0.0437, 0.4929, 0.3082, 0.0818, 0.0734],
         [0.0718, 0.5613, 0.0633, 0.2867, 0.0169]],

        [[0.1545, 0.4207, 0.0120, 0.2460, 0.1668],
         [0.1221, 0.0251, 0.0200, 0.8237, 0.0091]],

        [[0.1545, 0.4207, 0.0120, 0.2460, 0.1668],
         [0.1221, 0.0251, 0.0200, 0.8237, 0.0091]],

        [[0.1545, 0.4207, 0.0120, 0.2460, 0.1668],
         [0.1221, 0.0251, 0.0200, 0.8237, 0.0091]],

        [[0.0707, 0.2742, 0.2635, 0.1047, 0.2869],
         [0.1221, 0.0251, 0.0200, 0.8237, 0.0091]],

        [[0.0707, 0.2742, 0.2635, 0.1047, 0.2869],
         [0.3246, 0.2208, 0.1100, 0.2588, 0.0856]],

        [[0.6231, 0.0761, 0.2051, 0.0722, 0.0235],
         [0.3246, 0.2208, 0.1100, 0.2588, 0.0856]]], grad_fn=<ExpBackward0>)
```

### Forward Pass with Custom Loss

#### Negative log-likelihood ``nll``
```py
>>> crf(emissions, tags, mask=mask)
tensor(209.5386, grad_fn=<MeanBackward0>)
```

#### Corrected negative log-likelihood ``c_nll``
```py
>>> crf(emissions, tags, mask=mask, loss_fn="c_nll")
tensor(618.7924, grad_fn=<MeanBackward0>)
```

#### Generalized cross-entropy ``gce``
```py
>>> crf(emissions, tags, mask=mask, loss_fn="gce")
tensor(1.0149, grad_fn=<MeanBackward0>)
```



# License

MIT



# References

> kajyuuen. pytorch-partial-crf . 2021. [GitHub Repository](https://github.com/kajyuuen/pytorch-partial-crf/tree/master)

> yumeng5. RoSTER. 2021. [GitHub Repository](https://github.com/yumeng5/RoSTER/tree/main)

> amzn. amazon-weak-ner-needle. 2023. [GitHub Repository](https://github.com/amzn/amazon-weak-ner-needle)