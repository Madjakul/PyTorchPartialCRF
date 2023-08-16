# pytorch_partial_crf/partial_crf.py

from typing import Optional, Literal, Union

import torch

from pytorch_partial_crf.base_crf import BaseCRF
from pytorch_partial_crf.utils import (
    IMPOSSIBLE_SCORE, create_possible_tag_masks
)



class PartialCRF(BaseCRF):
    """Partial/Fuzzy Conditional random field.

    Parameters
    ----------
    q: float, optional
        Hyperparameter used to modify the generalized cross-entropy.

    Attributes
    ----------
    q: float, default=0.7
        Hyperparameter used to modify the generalized cross-entropy.
    """

    def __init__(
        self, num_tags: int, device: Literal["cpu", "cuda"],
        q: Optional[float]=None, padding_idx: Optional[int]=None
    ) -> None:
        super().__init__(num_tags, device, padding_idx)
        self.q = q if q is not None else 0.7

    def _numerator_score(
        self, emissions: Union[torch.FloatTensor, torch.cuda.FloatTensor],
        mask: Union[torch.ByteTensor, torch.cuda.ByteTensor],
        possible_tags: Union[torch.ByteTensor, torch.cuda.ByteTensor]
    ) -> Union[torch.ByteTensor, torch.cuda.FloatTensor]:
        """
        Computes the log of the emission/unary score plus the transition score
        for the whole sequence.

        Parameters
        ----------
        emissions: Union[torch.FloatTensor, torch.cuda.FloatTensor]
            Unary/emission score of each tokens.
            (batch_size, sequence_length, num_tags).
        mask: Union[torch.ByteTensor, torch.cuda.ByteTensor]
            Masked used to to discard subwords, special tokens or padding from
            being added to the log-probability. (batch_size, sequence_length).
        possible_tags: Union[torch.ByteTensor, torch.cuda.ByteTensor]
            Mask corresponding to the target label(s).
            (batch_size, sequence_length, num_tags).

        Returns
        -------
        Union[torch.FloatTensor, torch.cuda.FloatTensor]
            Log probability of the emission/unary score plus the transition
            score for the whole sequence. (batch_size,)
        """
        batch_size, sequence_length, num_tags = emissions.data.shape
        emissions = emissions.transpose(0, 1).contiguous()
        mask = mask.float().transpose(0, 1).contiguous()
        possible_tags = possible_tags.float().transpose(0, 1)

        # Start transition score and first emission
        first_possible_tag = possible_tags[0]
        alpha = self.start_transitions + emissions[0]                       # (batch_size, num_tags)
        alpha[(first_possible_tag == 0)] = IMPOSSIBLE_SCORE

        for i in range(1, sequence_length):
            current_possible_tags = possible_tags[i-1]                      # (batch_size, num_tags)
            next_possible_tags = possible_tags[i]                           # (batch_size, num_tags)

            # Emissions scores
            emissions_score = emissions[i]
            emissions_score[(next_possible_tags == 0)] = IMPOSSIBLE_SCORE
            emissions_score = emissions_score.view(batch_size, 1, num_tags)

            # Transition scores
            transition_scores = self.transitions.unsqueeze(0).expand(
                batch_size, num_tags, num_tags
            ).clone()
            transition_scores[(current_possible_tags == 0)] = \
                IMPOSSIBLE_SCORE
            transition_scores.transpose(1, 2)[(next_possible_tags == 0)] = \
                IMPOSSIBLE_SCORE

            # Broadcast alpha
            broadcast_alpha = alpha.unsqueeze(2)

            # Add all scores
            inner = broadcast_alpha + emissions_score + transition_scores   # (batch_size, num_tags, num_tags)
            alpha = (
                torch.logsumexp(inner, 1) * mask[i].unsqueeze(1)
                + alpha * (1 - mask[i]).unsqueeze(1)
            )

        # Add end transition score
        last_tag_indexes = mask.sum(0).long() - 1
        end_transitions = (
            self.end_transitions.expand(batch_size, num_tags)
            * possible_tags.transpose(0, 1).view(
                sequence_length * batch_size, num_tags
            )[
                last_tag_indexes
                + torch.arange(batch_size, device=possible_tags.device)
                * sequence_length
            ]
        )
        end_transitions[(end_transitions == 0)] = IMPOSSIBLE_SCORE
        stops = alpha + end_transitions                                     # (batch_size, num_tags)
        return torch.logsumexp(stops, 1)                                    # (batch_size,)

    def _denominator_score(
        self, emissions: Union[torch.FloatTensor, torch.cuda.FloatTensor],
        mask: Union[torch.ByteTensor, torch.cuda.ByteTensor],
    ) -> Union[torch.FloatTensor, torch.cuda.FloatTensor]:
        """
        Computes the log-partition score for the whole sequence.

        Parameters
        ----------
        emissions: Union[torch.FloatTensor, torch.cuda.FloatTensor]
            Unary/emission score of each tokens.
            (batch_size, sequence_length, num_tags).
        mask: Union[torch.ByteTensor, torch.cuda.ByteTensor]
            Masked used to to discard subwords, special tokens or padding from
            being added to the log-probability. (batch_size, sequence_length).

        Returns
        -------
        Union[torch.FloatTensor, torch.cuda.FloatTensor]
            Log-partition score. (batch_size,)
        """
        _, sequence_length, _ = emissions.data.shape
        emissions = emissions.transpose(0, 1).contiguous()
        mask = mask.float().transpose(0, 1).contiguous()
        # Start transition score and first emissions score
        alpha = self.start_transitions.unsqueeze(0) + emissions[0]

        for i in range(1, sequence_length):
            emissions_score = emissions[i].unsqueeze(1)                     # (batch_size, 1, num_tags)
            transition_scores = self.transitions.unsqueeze(0)               # (1, num_tags, num_tags)
            broadcast_alpha = alpha.unsqueeze(2)                            # (batch_size, num_tags, 1)
            inner = broadcast_alpha + emissions_score + transition_scores   # (batch_size, num_tags, num_tags)
            alpha = (
                torch.logsumexp(inner, 1) * mask[i].unsqueeze(1)
                + alpha * (1 - mask[i]).unsqueeze(1)
            )

        # Add end transition score
        stops = alpha + self.end_transitions.unsqueeze(0)
        return torch.logsumexp(stops, 1)                                    # (batch_size,)

    def forward(
        self, emissions: Union[torch.FloatTensor, torch.cuda.FloatTensor],
        tags: Union[torch.LongTensor, torch.cuda.LongTensor],
        loss_fn: Literal["nll", "c_nll", "gce"]="nll",
        mask: Optional[Union[torch.ByteTensor, torch.cuda.ByteTensor]]=None
    ) -> Union[torch.FloatTensor, torch.cuda.FloatTensor]:
        """Performs the forward pass depending on the loss function chosen: the
        classic negative log-likelihood, the corrected negative log-likelihood
        where the the negative log-unlikelihood [3]_ is computed and used as a
        regularizer [4]_ and the generelized cross-entropy [5]_ [6]_.

        Parameters
        ----------
        emissions: Union[torch.FloatTensor, torch.cuda.FloatTensor]
            Unary/emission score of each tokens.
            (batch_size, sequence_length, num_tags).
        tags: Union[torch.LongTensor, torch.cuda.LongTensor]
            Tensor containing the target labels. (batch_size, sequence_length).
        loss_fn: str, {"nll", "c_nll", "gce"}, default="nll"
            Loss function to use: "nll" for negative log-likelihood, "c_nll"
            for corrected negative log-likelihood or "gce" for generalized
            cross-entropy.
        mask: Union[torch.ByteTensor, torch.cuda.ByteTensor], optional
            Masked used to to discard subwords, special tokens or padding from
            being added to the log-probability. (batch_size, sequence_length).

        Returns
        -------
        Union[torch.FloatTensor, torch.cuda.FloatTensor]
            Mean of the losses over the mini-batch. (0,)

        References
        ----------
        ..  [3] Welleck, Sean, et al. "Neural text generation with unlikelihood
                training." arXiv preprint arXiv:1908.04319 (2019).
        ..  [4] Jiang, Haoming, et al. "Named entity recognition with small
                strongly labeled and large weakly labeled data." arXiv preprint
                arXiv:2106.08977 (2021).
        ..  [5] Zhang, Zhilu, and Mert Sabuncu. "Generalized cross entropy loss
                for training deep neural networks with noisy labels." Advances
                in neural information processing systems 31 (2018).
        ..  [6] Meng, Yu, et al. "Distantly-supervised named entity recognition
                with noise-robust learning and language model augmented
                self-training." arXiv preprint arXiv:2109.05003 (2021).
        """
        if mask is None:
            mask = torch.ones_like(tags, dtype=torch.uint8)
        possible_tags = create_possible_tag_masks(self.num_tags, tags)          # (batch_size, sequence_length, num_tags)
        pred = self.marginal_probabilities(emissions, mask).transpose(0, 1)     # (batch_size, sequence_length, num_tags)
        p_mask = possible_tags.eq(1.0)                                          # (batch_size, sequence_length, num_tags)
        p_bar_mask = possible_tags.eq(0.0)                                      # (batch_size, sequence_length, num_tags)
        p = torch.masked_select(pred, p_mask)                                   # (possible_tags==1,)
        p_bar = torch.masked_select(pred, p_bar_mask)                           # (possible_tags==0,)

        if loss_fn in ("nll", "c_nll"):
            gold_score = self._numerator_score(emissions, mask, possible_tags)  # (batch_size,)
            forward_score = self._denominator_score(emissions, mask)            # (batch_size,)
            nll = forward_score - gold_score                                    # (batch_size,)
            if loss_fn == "nll":
                return torch.mean(nll)                                          # Mean instead of sum
            nlu = -(1 - (-nll).exp()).log()
            if torch.isnan(nlu).any() or torch.isinf(nlu).any():
                nl = (1 - (-nll).exp())
                nl = nl + (nl < 1e-4).to(nl).detach() * (1e-4 - nl).detach()
                nlu = - nl.log()
            c_nll = torch.mean(p) * nll + torch.mean(p_bar) * nlu               # Mean because it's the expecteancy (?)
            return torch.mean(c_nll)
        gce = (1 - p**self.q) / self.q
        gce = torch.mean(gce)                                                   # (loss.view(-1)*weights).sum() / weights.sum()
        return gce

