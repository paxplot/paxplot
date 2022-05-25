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


def modelevaluation():
    """
    Model evaluation dataset

    Returns
    -------
    stream : _io.BufferedReader
        Stream of trade-off dataset
    """
    stream = pkg_resources.resource_stream(__name__, 'data/modelevaluation.csv')
    return stream
