
import yfinance as yf
import requests_cache
from requests import Session
from requests.adapters import HTTPAdapter
from urllib3.util import Retry

def get_stable_session():
    session = Session()
    # Mask as a real browser
    session.headers.update({
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    })
    # Add retry logic for network stability
    retries = Retry(total=3, backoff_factor=1, status_forcelist=[502, 503, 504])
    session.mount('https://', HTTPAdapter(max_retries=retries))
    return session


def fetch_market_data(ticker="SPY", start_date=None, end_date=None, period="35y"):
    session= get_stable_session()
    if start_date and end_date:
        raw_data = yf.download([ticker, "^VIX"], start=start_date, end=end_date, session=session, progress=False)
    else:
        raw_data = yf.download([ticker, "^VIX"], period=period, session=session, progress=False)

    # Extract ticker data
    df = raw_data.xs(ticker, level=1, axis=1).copy()
    
    # Extract VIX Close
    vix_close = raw_data.xs('^VIX', level=1, axis=1)['Close']
    df['VIX'] = vix_close

    df['VIX'] = df['VIX'].ffill() 
    
    df['VIX_Safe'] = (df['VIX'] < 25).astype(int)
    return df.dropna()