import FinanceDataReader as fdr
import matplotlib.pyplot as plt
from datetime import datetime

class Crawler:
    def __init__(self, symbol:str):
        """_summary_
        Crawler 객체를 생성, 사용자가 지정한 주식 심볼에 대한 데이터 리더
        
        Args:
            symbol (str): 주식 심볼
        """
        self.symbol = symbol  # 사용자가 입력한 주식 심볼을 객체의 심볼로 설정
        self.end_date = datetime.now().strftime('%Y-%m-%d') 
        print(f"Crawler instance created for stock symbol: {symbol}")

    def get_stock_data(self, start_date:str, end_date:str):
        """_summary_
        객체에 저장된 주식 심볼의 데이터를 지정된 날짜 범위 동안 가져옴
        
        Args:
            start_date (str): 시작 날짜 ('YYYY-MM-DD' 형식)
            end_date (str): 종료 날짜 ('YYYY-MM-DD' 형식)
        Returns:
            pd.DataFrame : 날짜 범위동안의 주식 데이터(시가, 최고가, 최저가, 종가, 거래량)
        """
        data = fdr.DataReader(self.symbol, start_date, end_date)
        return data

    def get_exchange_rate(self, currency_symbol:str, start_date:str, end_date:str):
        """_summary_
        지정된 화폐 교환 심볼에 대한 환율 데이터를 지정된 날짜 범위 동안 가져옴

        Args:
            currency_symbol (str): 화폐 교환 심볼 (예: 'USD/KRW')
            start_date (str): 시작 날짜 ('YYYY-MM-DD' 형식)
            end_date (str): 종료 날짜 ('YYYY-MM-DD' 형식)
        Returns:
            pd.DataFrame : 날짜 범위동안의 환율 데이터
        """
        data = fdr.DataReader(currency_symbol, start_date, end_date=self.end_date)
        return data
    
    def get_data_for_date(self, date:str):
        """_summary_
        지정된 특정일의 주식 데이터를 반환
        
        Args:
            date (str): 데이터를 조회할 특정 날짜 ('YYYY-MM-DD' 형식)
        Returns:
            _type_: 해당 날짜의 DataFrame 행
        """
        data = self.get_stock_data(date, date)
        if not data.empty:
            print(f"Data for {self.symbol} on {date}:")
            print(f"Open: {data['Open'].iloc[0]}")
            print(f"High: {data['High'].iloc[0]}")
            print(f"Low: {data['Low'].iloc[0]}")
            print(f"Close: {data['Close'].iloc[0]}")
            print(f"Volume: {data['Volume'].iloc[0]}")
            return data.iloc[0]  # 데이터가 있는 경우 해당 날짜의 첫 번째 행 반환
        else:
            return None  # 데이터가 없는 경우 None 반환
    
    def calculate_return(self, start_date:str, end_date:str):
        """_summary_
        지정된 날짜 범위 동안의 주식의 수익률을 계산
        
        Args:
            start_date (str): 수익률 계산을 시작할 날짜
            end_date (str): 수익률 계산을 종료할 날짜
        Returns:
            float : 해당 기간 동안의 수익률 (퍼센트 단위)
        """
        data = self.get_stock_data(start_date, end_date)
        start_price = data['Close'].iloc[0]
        end_price = data['Close'].iloc[-1]
        return_rate = ((end_price - start_price) / start_price) * 100
        print(f"The return rate from {start_date} to {end_date} is {return_rate:.2f}%")
        return return_rate

    def calculate_correlation(self, other_symbol:str, start_date:str, end_date:str):
        """_summary_
        지정된 종목과 다른 종목 간의 일간 수익률의 상관관계를 계산
        
        Args:
            other_symbol (str): 비교할 다른 주식의 심볼
            start_date (str): 데이터 시작 날짜
            end_date (str): 데이터 종료 날짜
        Returns:
            float : 상관계수
        """
        data1 = self.get_stock_data(start_date, end_date)
        data2 = Crawler(other_symbol).get_stock_data(start_date, end_date)

        # 일간 수익률 계산
        returns1 = data1['Close'].pct_change().dropna()
        returns2 = data2['Close'].pct_change().dropna()

        # 상관계수 계산
        correlation = returns1.corr(returns2)
        return correlation
    
    def plot_stock_data(self, start_date:str, end_date:str):
        """_summary_
        지정된 날짜 범위 동안의 주식 가격 데이터를 시각화
        
        Args:
            start_date (str): 그래프를 시작할 날짜
            end_date (str): 그래프를 종료할 날짜
        """
        data = self.get_stock_data(start_date, end_date)
        plt.figure(figsize=(10, 6))
        plt.plot(data.index, data['Close'], label='Close Price')
        plt.title(f'Stock Price of {self.symbol} from {start_date} to {end_date}')
        plt.xlabel('Date')
        plt.ylabel('Close Price')
        plt.legend()
        plt.grid(True)
        plt.show()
    
    def list_stocks(self, market='KRX'):
        """_summary_
        지정된 주식 시장의 종목 리스트를 출력
        
        Args:
            market (str, optional): 주식 시장 심볼('KRX', 'NASDAQ', 'NYSE', 'S&P500', 등). Defaults to 'KRX'.
        Returns:
            pd.DataFrame : 주식 시장 전체 종목 정보
        """
        stock_dataframe = fdr.StockListing(market)
        if stock_dataframe is not None:
            return stock_dataframe
        else:
            print(f"No data available for the {market} market")
