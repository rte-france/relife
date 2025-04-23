from relife.lifetime_model import AgeReplacementModel


def test_distribution(distribution, time):
    ar_model = AgeReplacementModel(distribution)

    assert ar_model.args_names == ("ar",)


    frozen_model = ar_model.freeze(1)
    frozen_model.sf(time())

    assert ar_model.sf(time(), 1.).shape == ()

