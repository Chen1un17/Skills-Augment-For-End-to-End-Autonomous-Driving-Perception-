from ad_cornercase.edge.uncertainty import normalized_entropy
from ad_cornercase.schemas.common import CandidateLabel


def test_normalized_entropy_is_low_for_confident_distribution() -> None:
    value = normalized_entropy(
        [
            CandidateLabel(label="truck", probability=0.95),
            CandidateLabel(label="debris", probability=0.03),
            CandidateLabel(label="sign", probability=0.02),
        ]
    )
    assert value < 0.3


def test_normalized_entropy_is_high_for_flat_distribution() -> None:
    value = normalized_entropy(
        [
            CandidateLabel(label="truck", probability=0.34),
            CandidateLabel(label="debris", probability=0.33),
            CandidateLabel(label="sign", probability=0.33),
        ]
    )
    assert value > 0.95
