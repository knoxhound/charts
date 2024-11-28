import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
from typing import List, Tuple
import requests


class CryptoVisualizer:
    def __init__(self, symbols: List[str], timeframe: str = '1m', limit: int = 100):
        """
        Initialize the CryptoVisualizer with specified cryptocurrency symbols.

        Args:
            symbols: List of cryptocurrency IDs (e.g., ['sei', 'sui', 'ripple', 'ethereum'])
            timeframe: Candlestick timeframe (default: '1m')
            limit: Number of historical candles to fetch (default: 100)
        """
        self.base_url = "https://api.coingecko.com/api/v3"
        self.symbols = symbols
        self.timeframe = timeframe
        self.limit = limit
        self.figures = {}

    def fetch_ohlcv_data(self, symbol: str) -> pd.DataFrame:
        """Fetch OHLCV data for a given symbol using CoinGecko API."""
        try:
            # Convert timeframe to days (CoinGecko uses days for historical data)
            days = min(max(self.limit, 1), 90)  # CoinGecko free tier limits to 90 days

            # Construct the API URL
            url = f"{self.base_url}/coins/{symbol}/ohlc"
            params = {
                'vs_currency': 'usd',
                'days': days
            }

            # Make the API request
            response = requests.get(url, params=params)
            response.raise_for_status()  # Raise an exception for bad status codes
            data = response.json()

            # Convert the data to a DataFrame
            df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')

            # Add volume data if available (Note: CoinGecko's free tier doesn't provide volume in OHLC)
            df['volume'] = np.nan

            return df

        except Exception as e:
            print(f"Error fetching data for {symbol}: {str(e)}")
            return pd.DataFrame()

    def calculate_bollinger_bands(self, df: pd.DataFrame, window: int = 20, num_std: float = 2) -> Tuple[
        pd.Series, pd.Series]:
        """Calculate Bollinger Bands for the given DataFrame."""
        sma = df['close'].rolling(window=window).mean()
        std = df['close'].rolling(window=window).std()
        upper_band = sma + (std * num_std)
        lower_band = sma - (std * num_std)
        return upper_band, lower_band

    def calculate_rsi(self, df: pd.DataFrame, periods: int = 14) -> pd.Series:
        """Calculate RSI (Relative Strength Index)."""
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=periods).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=periods).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    def create_visualization(self, symbol: str, df: pd.DataFrame) -> go.Figure:
        """Create a candlestick chart with technical indicators."""
        if df.empty:
            return None

        # Calculate technical indicators
        upper_band, lower_band = self.calculate_bollinger_bands(df)
        rsi = self.calculate_rsi(df)

        # Create subplots for price action and RSI
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                            vertical_spacing=0.03,
                            row_heights=[0.7, 0.3])

        # Add candlestick chart
        fig.add_trace(go.Candlestick(
            x=df['timestamp'],
            open=df['open'],
            high=df['high'],
            low=df['low'],
            close=df['close'],
            name='Price'
        ), row=1, col=1)

        # Add Bollinger Bands
        fig.add_trace(go.Scatter(
            x=df['timestamp'],
            y=upper_band,
            name='Upper BB',
            line=dict(color='gray', dash='dash')
        ), row=1, col=1)

        fig.add_trace(go.Scatter(
            x=df['timestamp'],
            y=lower_band,
            name='Lower BB',
            line=dict(color='gray', dash='dash'),
            fill='tonexty'
        ), row=1, col=1)

        # Add RSI
        fig.add_trace(go.Scatter(
            x=df['timestamp'],
            y=rsi,
            name='RSI',
            line=dict(color='purple')
        ), row=2, col=1)

        # Add RSI levels
        fig.add_hline(y=70, line_color='red', line_dash='dash', row=2, col=1)
        fig.add_hline(y=30, line_color='green', line_dash='dash', row=2, col=1)

        # Update layout
        fig.update_layout(
            title=f'{symbol.upper()} Real-time Chart',
            xaxis_title='Time',
            yaxis_title='Price (USD)',
            yaxis2_title='RSI',
            xaxis_rangeslider_visible=False,
            template='plotly_dark'  # Use dark theme for better visibility
        )

        return fig

    def update_visualization(self):
        """Update visualizations for all symbols in real-time."""
        while True:
            try:
                for symbol in self.symbols:
                    print(f"Fetching data for {symbol}...")
                    df = self.fetch_ohlcv_data(symbol)
                    if not df.empty:
                        fig = self.create_visualization(symbol, df)
                        if fig:
                            fig.show()

                    # Add delay between requests to respect API rate limits
                    time.sleep(2)

                # Wait before next update cycle
                print("Waiting for next update cycle...")
                time.sleep(60)  # Update every minute

            except Exception as e:
                print(f"Error updating visualization: {str(e)}")
                time.sleep(5)

    def get_available_coins(self) -> List[str]:
        """Get list of available coins from CoinGecko."""
        try:
            response = requests.get(f"{self.base_url}/coins/list")
            response.raise_for_status()
            coins = response.json()
            return [(coin['id'], coin['symbol'].upper(), coin['name']) for coin in coins]
        except Exception as e:
            print(f"Error fetching available coins: {str(e)}")
            return []


def main():
    # Define cryptocurrency symbols to track (using CoinGecko IDs)
    symbols = [
        'sei-network',  # SEI
        'sui',  # SUI
        'ripple',  # XRP
        'ethereum'  # ETH
    ]

    # Initialize and run the visualizer
    visualizer = CryptoVisualizer(symbols)

    # Optional: Print available coins
    print("Fetching available coins...")
    available_coins = visualizer.get_available_coins()
    print(f"Total available coins: {len(available_coins)}")
    print("\nStarting real-time cryptocurrency visualization...")

    visualizer.update_visualization()


if __name__ == "__main__":
    main()