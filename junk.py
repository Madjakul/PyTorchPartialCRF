import torch

from pytorch_partial_crf import PartialCRF


emissions = torch.randn(2, 5, 5)
mask = torch.ByteTensor([
    [0, 1, 1, 1, 0],
    [1, 1, 1, 0, 0]
])
tags = torch.LongTensor([
    [1, 2, -1, 3, 4],
    [-1, 3, 3, 2, -1],
])
crf = PartialCRF(5, device="cpu")
# print(crf.marginal_probabilities(emissions, mask), "\n\n")
# print(type(crf.viterbi_decode(emissions, mask)), "\n\n")
print(crf(emissions, tags, mask))
