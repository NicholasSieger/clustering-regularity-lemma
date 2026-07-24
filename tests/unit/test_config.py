import pytest

from clustering_regularity.config import PaperParameters, RunConfig


def test_paper_thresholds():
    parameters = PaperParameters(0.05)

    assert parameters.delta_1 == pytest.approx(0.05**2 / 9)
    assert parameters.delta_2 == pytest.approx(2 * 0.05**2 / 5)
    assert parameters.delta_3 == pytest.approx(0.05**5 / 90)
    assert parameters.delta_4 == pytest.approx(0.05 ** (5 / 2) / 9)
    assert parameters.delta_5 == pytest.approx(2 * 0.05 ** (5 / 2) / 5)
    assert parameters.delta_6 == pytest.approx(0.05**5)


@pytest.mark.parametrize("epsilon", (0.0, 1 / 16, 0.1))
def test_paper_domain_is_enforced(epsilon):
    with pytest.raises(ValueError, match="epsilon < 1/16"):
        PaperParameters(epsilon)


def test_max_rounds_is_an_execution_limit():
    config = RunConfig(PaperParameters(), max_rounds=4)

    assert config.max_rounds == 4
    assert config.parameters.epsilon == 0.05

