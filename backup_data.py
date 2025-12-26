import yfinance as yf

# Download the last 30+ years of data
print("Downloading historical backup...")
data = yf.download(["SPY", "^VIX"], period="max")

# Save it to your project folder
data.to_csv("historical_backup.csv")
print("✅ historical_backup.csv created! Now push this file to GitHub.")