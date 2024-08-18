import numpy as np
import pandas as pd

from stock_advisor.crawler import Crawler


def test_get_data(samsung_crawler: Crawler):
    data = samsung_crawler.get_stock_data("2023-01-02", "2023-04-28")

    assert data.index[0] == pd.Timestamp("2023-01-02")
    assert data.index[-1] == pd.Timestamp("2023-04-28")


def test_calculate_return(samsung_crawler: Crawler):
    return_value = samsung_crawler.calculate_return("2023-01-02", "2023-04-28")

    assert np.isclose(18.01801801801802, return_value)


def test_calcuate_correlation(samsung_crawler: Crawler):
    correlation = samsung_crawler.calculate_correlation("000660", "2023-01-02", "2023-04-28")

    assert np.isclose(0.7769778586545224, correlation)
