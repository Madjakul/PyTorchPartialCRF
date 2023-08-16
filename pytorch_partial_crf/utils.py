# pytorch_partial_crf/utils.py

import torch


UNLABELED_INDEX = -1
IMPOSSIBLE_SCORE = -100


def create_possible_tag_masks(
    num_tags: int, tags: torch.LongTensor,
) -> torch.ByteTensor:
    """Creates a mask-like sparse tensor where the index of the correct tag has
    a value of 1, allowing for multilabel targets.

    Parameters
    ----------
    num_tags: int
        Number of different tags in the dataset.
    tags: torch.LongTensor
        Tensor of target labels. (batch_size, sequence_length).

    Returns
    -------
    masks: torch.ByteTensor
        Mask-like sparse tensor indicating the target label.
        (batch_size, sequence_length, num_tags).
    """
    copy_tags = tags.clone()
    no_annotation_idx = (copy_tags == UNLABELED_INDEX)
    copy_tags[copy_tags == UNLABELED_INDEX] = 0

    tags_ = torch.unsqueeze(copy_tags, 2)
    masks = torch.zeros(
        tags_.size(0),
        tags_.size(1),
        num_tags,
        dtype=torch.uint8,
        device=tags.device
    )
    masks.scatter_(2, tags_, 1)
    masks[no_annotation_idx] = 1
    return masks    # type: ignore

