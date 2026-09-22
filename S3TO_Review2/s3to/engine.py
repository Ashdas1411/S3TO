from dataclasses import dataclass
from typing import Dict

import numpy as np

from .grouping import assignment_from_groups, choose_complementary_groups


@dataclass
class S3TOResult:
    histograms: Dict[int, list]
    groups: list
    group_scores: list
    assignment: Dict[int, str]
    pairwise_cosine: dict


class S3TOEngine:
    def __init__(self, group_size=3):
        self.group_size = group_size

    def decide(self, histograms):
        clean = {int(k): np.asarray(v, dtype=float) for k, v in histograms.items()}
        groups, scores, pairwise = choose_complementary_groups(clean, self.group_size)
        return S3TOResult(
            histograms={k: v.tolist() for k, v in clean.items()},
            groups=groups,
            group_scores=[float(x) for x in scores],
            assignment=assignment_from_groups(groups),
            pairwise_cosine=pairwise,
        )
