"""Default datasets included with Paxplot"""

import pkg_resources


def tradeoff():
    """
    Trade-off dataset

    Returns
    -------
    stream : _io.BufferedReader
        Stream of trade-off dataset
    """
    stream = pkg_resources.resource_stream(__name__, 'data/tradeoff.csv')
    return stream


def hydroclimate_model_evaluation():
    """
    Hydroclimate model evaluation dataset from Nele Reyniers

    Returns
    -------
    stream : _io.BufferedReader
        Stream of trade-off dataset
    """
    stream = pkg_resources.resource_stream(
        __name__,
        'data/hydroclimate_model_evaluation.csv'
    )
    return stream
