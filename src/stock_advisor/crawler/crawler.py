from datetime import datetime
from typing import Optional

import FinanceDataReader as fdr
import matplotlib.pyplot as plt
import pandas as pd


class Crawler:
    def __init__(self, symbol: str) -> None:
        """initiate Crawer object.

        Args:
            symbol (str): stock symbol.
        """
        self.symbol = symbol
        self.end_date = datetime.now().strftime("%Y-%m-%d")
        print(f"Crawler instance created for stock symbol: {symbol}")

    @classmethod
    def get_crawler(cls, symbol: str) -> "Crawler":
        """get Crawler object.

        Args:
            symbol (str): stock symbol.

        Returns:
            Crawler: Crawler of symbol.
        """
        return cls(symbol)

    def get_stock_data(self, start_date: str, end_date: Optional[str] = None) -> pd.DataFrame:
        """get stock data from saved stock of symbol.
        from start_date to end_date, get stock data.

        Args:
            start_date (str): start date ('YYYY-MM-DD')
            end_date (Optional[torch.Tensor], optional): end date ('YYYY-MM-DD')

        Returns:
            pd.DataFrame: stock data.
        """
        data = fdr.DataReader(self.symbol, start_date, self.end_date if end_date is None else end_date)
        return data

    def get_exchange_rate(self, currency_symbol: str, start_date: str, end_date: Optional[str] = None) -> pd.DataFrame:
        """get exchange rate from currency_symbol.
        from start_date to end_date, get exchange rate of currency_symbol.

        Args:
            currency_symbol (str): currency exchange symbol (e.g. 'USD/KRW')
            start_date (str): start date ('YYYY-MM-DD')
            end_date (Optional[torch.Tensor], optional): end date ('YYYY-MM-DD')

        Returns:
            pd.DataFrame : exchange raet data.
        """
        data = fdr.DataReader(
            currency_symbol,
            start_date,
            end_date=self.end_date if end_date is None else end_date,
        )
        return data

    def get_data_for_date(self, date: str) -> Optional[pd.Series]:
        """get date of input date.

        Args:
            date (str): date to search. ('YYYY-MM-DD')

        Returns:
            Optional[pd.Series]: data of input date. if cannot get data, return None.
        """
        data = self.get_stock_data(date, date)
        if not data.empty:
            print(f"Data for {self.symbol} on {date}:")
            print(f"Open: {data['Open'].iloc[0]}")
            print(f"High: {data['High'].iloc[0]}")
            print(f"Low: {data['Low'].iloc[0]}")
            print(f"Close: {data['Close'].iloc[0]}")
            print(f"Volume: {data['Volume'].iloc[0]}")
            return data.iloc[0]
        else:
            return None

    def calculate_return(self, start_date: str, end_date: Optional[str] = None) -> float:
        """from star_date to end_date, calculate return.

        Args:
            start_date (str): starte date.
            end_date (Optional[torch.Tensor], optional): end date.

        Returns:
            float : calcuated return.
        """
        data = self.get_stock_data(start_date, end_date)
        start_price = data["Close"].iloc[0]
        end_price = data["Close"].iloc[-1]
        return_rate = ((end_price - start_price) / start_price) * 100
        print(f"The return rate from {start_date} to {end_date} is {return_rate:.2f}%")
        return return_rate

    def calculate_correlation(self, other_symbol: str, start_date: str, end_date: Optional[str] = None) -> float:
        """calcuate correlation between object's symbol and other symbol.

        Args:
            other_symbol (str): symbol of compared stock.
            start_date (str): start date.
            end_date (Optional[torch.Tensor], optional): end date.

        Returns:
            float : correlation value.
        """
        end_date = self.end_date if end_date is None else end_date
        data1 = self.get_stock_data(start_date, end_date)
        data2 = Crawler.get_crawler(other_symbol).get_stock_data(start_date, end_date)

        returns1 = data1["Close"].pct_change().dropna()
        returns2 = data2["Close"].pct_change().dropna()

        correlation = returns1.corr(returns2)
        return correlation

    def plot_stock_data(self, start_date: str, end_date: Optional[str] = None) -> None:
        """plot close price of stock.

        Args:
            start_date (str): start date.
            end_date (Optional[torch.Tensor], optional): end date.
        """
        data = self.get_stock_data(start_date, end_date)
        plt.figure(figsize=(10, 6))
        plt.plot(data.index, data["Close"], label="Close Price")
        plt.title(f"Stock Price of {self.symbol} from {start_date} to {end_date}")
        plt.xlabel("Date")
        plt.ylabel("Close Price")
        plt.legend()
        plt.grid(True)
        plt.show()

    def list_stocks(self, market: str = "KRX") -> pd.DataFrame:
        """get stock list of market.

        Args:
            market (str): market symbol.(e.g. 'KRX', 'NASDAQ', 'NYSE', 'S&P500'). Defaults to 'KRX'.

        Returns:
            pd.DataFrame : market stock list.
        """
        stock_dataframe = fdr.StockListing(market)
        if stock_dataframe is not None:
            return stock_dataframe
        else:
            raise ValueError(f"No data available for the {market} market")
