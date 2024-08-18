import pytest

from stock_advisor.crawler import Crawler


@pytest.fixture
def samsung_crawler() -> Crawler:
    return Crawler.get_crawler("005930")
