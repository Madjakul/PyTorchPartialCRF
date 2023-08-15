from typing import Optional, Literal, Union

import torch

from pytorch_partial_crf.base_crf import BaseCRF
from pytorch_partial_crf.utils import create_possible_tag_masks

from pytorch_partial_crf.utils import IMPOSSIBLE_SCORE


class PartialCRF(BaseCRF):
    """Partial/Fuzzy Conditional random field.
    """
    __doc__ = BaseCRF.__doc__ + __doc__

    def __init__(
        self, num_tags: int, device: Literal["cpu", "cuda"],
        padding_idx: Optional[int]=None
    ) -> None:
        super().__init__(num_tags, device, padding_idx)

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
        mask: Optional[Union[torch.ByteTensor, torch.cuda.ByteTensor]]=None
    ) -> Union[torch.FloatTensor, torch.cuda.FloatTensor]:
        if mask is None:
            mask = torch.ones_like(tags, dtype=torch.uint8)
        possible_tags = create_possible_tag_masks(self.num_tags, tags)
        print(possible_tags)
        p = self.marginal_probabilities(emissions, mask).transpose(0, 1)
        x = possible_tags.eq(1.0)
        new_p = torch.masked_select(p, x)
        print(new_p)
        loss = (1 - new_p**0.7) / 0.7
        print(loss)
        loss = loss.sum() / len(loss)
        print(loss)

        gold_score = self._numerator_score(emissions, mask, possible_tags)
        forward_score = self._denominator_score(emissions, mask)
        return torch.sum(forward_score - gold_score)

