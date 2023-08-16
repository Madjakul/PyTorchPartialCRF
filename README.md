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
tensor([[[ 0.9430, -0.9577, -0.7622, -0.8821, -0.5546],
         [ 0.2546,  0.2714,  1.5236,  1.0995,  0.3179],
         [-0.8178, -0.0739, -0.7091, -0.2723, -0.5086],
         [-1.2528, -0.5149, -1.0528,  0.2833, -0.3489],
         [-0.6397,  1.1646,  0.8108, -0.6258,  0.5861],
         [-0.3642, -1.1495,  0.4248,  0.9736,  0.6286],
         [-0.3671, -0.0908,  1.6077, -0.8238, -1.2533],
         [ 0.2673, -0.4499, -0.5565, -0.7856,  1.4061],
         [ 0.7015,  0.6881,  0.2957, -0.7657,  1.1108]],

        [[ 0.0447,  1.6346,  2.4109, -1.1235,  0.0994],
         [-2.3874,  0.1132, -0.7715, -0.9332, -0.7920],
         [-0.6172, -0.2362, -1.0649,  0.2438, -0.3730],
         [-0.2407, -0.8403,  0.7397, -0.0184, -0.1510],
         [-0.1634, -1.4326, -2.5147, -0.5119, -0.8874],
         [-0.7520,  0.9549, -0.4331, -0.2316,  1.0032],
         [-1.1907, -0.9092,  0.5498,  0.4893, -0.3429],
         [ 0.3192, -1.6791, -1.8147, -1.2304, -1.4591],
         [-1.4058, -0.8867,  0.0130, -1.5325, -0.6477]]])
>>>
>>> tags = torch.randint(0, 5, (batch_size, sequence_length))
>>> tags
tensor([[4, 0, -1, 4, 2, 4, 2, 4, 3],
        [3, 3, 1, 0, -1, 4, 4, 4, 1]])
>>>
>>> mask = torch.bernoulli(torch.empty(batch_size, sequence_length).uniform_(0, 1)).byte()
>>> mask
tensor([[0, 1, 1, 1, 1, 0, 0, 0, 1],
        [0, 0, 0, 1, 1, 1, 1, 1, 0]], dtype=torch.uint8)
```


## Using the Partial/Fuzzy CRF

### Decoding

```py
>>> crf = PartialCRF(num_tags, device="cpu")
>>>
>>> crf.viterbi_decode(emissions, mask)
[[0, 2, 4, 4, 1], [2, 2, 2, 4, 0]]
```

### Computing the Marginal Probabilities

```py
>>> crf.marginal_probabilities(emissions, mask)
tensor([[[0.7515, 0.0700, 0.1132, 0.0291, 0.0362],
         [0.0744, 0.2325, 0.6703, 0.0054, 0.0175]],

        [[0.0887, 0.1369, 0.3766, 0.2257, 0.1721],
         [0.0744, 0.2325, 0.6703, 0.0054, 0.0175]],

        [[0.1199, 0.2579, 0.1841, 0.1871, 0.2511],
         [0.0744, 0.2325, 0.6703, 0.0054, 0.0175]],

        [[0.0946, 0.1350, 0.1154, 0.3611, 0.2938],
         [0.0797, 0.0627, 0.3466, 0.1976, 0.3133]],

        [[0.0849, 0.2741, 0.2971, 0.0514, 0.2926],
         [0.3975, 0.0892, 0.0433, 0.1970, 0.2730]],

        [[0.0849, 0.2741, 0.2971, 0.0514, 0.2926],
         [0.0582, 0.3738, 0.0971, 0.0808, 0.3901]],

        [[0.0849, 0.2741, 0.2971, 0.0514, 0.2926],
         [0.0662, 0.0629, 0.2491, 0.3615, 0.2603]],

        [[0.0849, 0.2741, 0.2971, 0.0514, 0.2926],
         [0.5094, 0.2084, 0.0803, 0.1025, 0.0994]],

        [[0.1391, 0.4087, 0.1068, 0.0381, 0.3073],
         [0.5094, 0.2084, 0.0803, 0.1025, 0.0994]]], grad_fn=<ExpBackward0>)
```

### Forward Pass with Custom Loss

#### Negative log-likelihood ``nll``
```py
>>> crf(emissions, tags, mask=mask)
tensor(210.8574, grad_fn=<MeanBackward0>)
```

#### Corrected negative log-likelihood ``c_nll``
```py
>>> crf(emissions, tags, mask=mask, loss_fn="c_nll")
tensor(41.8532, grad_fn=<MeanBackward0>)
```

#### Generalized cross-entropy ``gce``
```py
>>> crf(emissions, tags, mask=mask, loss_fn="gce")
tensor(0.9967, grad_fn=<MeanBackward0>)
```



# License

MIT



# References

> kajyuuen. pytorch-partial-crf . 2021. [GitHub Repository](https://github.com/kajyuuen/pytorch-partial-crf/tree/master)

> yumeng5. RoSTER. 2021. [GitHub Repository](https://github.com/yumeng5/RoSTER/tree/main)

> amzn. amazon-weak-ner-needle. 2023. [GitHub Repository](https://github.com/amzn/amazon-weak-ner-needle)