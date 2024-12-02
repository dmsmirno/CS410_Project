from flask import Flask, request, jsonify, send_file, render_template
import matplotlib.pyplot as plt
import os
import asyncio
from datetime import datetime, timedelta
from autogen_agentchat.agents import CodingAssistantAgent, ToolUseAssistantAgent
from autogen_agentchat.teams import RoundRobinGroupChat, StopMessageTermination
from autogen_core.components.models import OpenAIChatCompletionClient
from autogen_core.components.tools import FunctionTool
import numpy as np
import pandas as pd
import yfinance as yf
from pytz import timezone

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('test.html')  # Ensure index.html is in the same directory

@app.route('/get', methods=["GET", "POST"])
async def chat():
    msg = request.form["msg"]
    input = msg
    print(input)
    result = await get_Chat_response(input)
    return result

def analyze_stock(ticker: str) -> dict:
    # import os
    # from datetime import datetime, timedelta

    # import matplotlib.pyplot as plt
    # import numpy as np
    # import pandas as pd
    # import yfinance as yf
    # from pytz import timezone  # type: ignore
    stock = yf.Ticker(ticker)
    end_date = datetime.now(timezone("UTC"))
    start_date = end_date - timedelta(days=365)
    hist = stock.history(start=start_date, end=end_date)
    if hist.empty:
        return {"error": "No historical data available for the specified ticker."}
    current_price = stock.info.get("currentPrice", hist["Close"].iloc[-1])
    year_high = stock.info.get("fiftyTwoWeekHigh", hist["High"].max())
    year_low = stock.info.get("fiftyTwoWeekLow", hist["Low"].min())

    # Calculate 50-day and 200-day moving averages
    ma_50 = hist["Close"].rolling(window=50).mean().iloc[-1]
    ma_200 = hist["Close"].rolling(window=200).mean().iloc[-1]

    # Calculate YTD price change and percent change
    ytd_start = datetime(end_date.year, 1, 1, tzinfo=timezone("UTC"))
    ytd_data = hist.loc[ytd_start:]  # type: ignore[misc]
    if not ytd_data.empty:
        price_change = ytd_data["Close"].iloc[-1] - ytd_data["Close"].iloc[0]
        percent_change = (price_change / ytd_data["Close"].iloc[0]) * 100
    else:
        price_change = percent_change = np.nan

    # Determine trend
    if pd.notna(ma_50) and pd.notna(ma_200):
        if ma_50 > ma_200:
            trend = "Upward"
        elif ma_50 < ma_200:
            trend = "Downward"
        else:
            trend = "Neutral"
    else:
        trend = "Insufficient data for trend analysis"

    # Calculate volatility (standard deviation of daily returns)
    daily_returns = hist["Close"].pct_change().dropna()
    volatility = daily_returns.std() * np.sqrt(252)  # Annualized volatility

    # Create result dictionary
    result = {
        "ticker": ticker,
        "current_price": current_price,
        "52_week_high": year_high,
        "52_week_low": year_low,
        "50_day_ma": ma_50,
        "200_day_ma": ma_200,
        "ytd_price_change": price_change,
        "ytd_percent_change": percent_change,
        "trend": trend,
        "volatility": volatility,
    }

    # Convert numpy types to Python native types for better JSON serialization
    for key, value in result.items():
        if isinstance(value, np.generic):
            result[key] = value.item()
    return result

async def get_Chat_response(text):
    stock_analysis_tool = FunctionTool(analyze_stock, description="Analyze stock data and generate a plot")
    stock_analysis_agent = ToolUseAssistantAgent(
        name="Stock_Analysis_Agent",
        model_client=OpenAIChatCompletionClient(model="gpt-4o-mini", api_key="Your API key"),
        registered_tools=[stock_analysis_tool],
        description="Analyze stock data and generate a plot",
        system_message="You are a helpful AI assistant. Solve tasks using your tools.",
    )
    report_agent = CodingAssistantAgent(
        name="Report_Agent",
        model_client=OpenAIChatCompletionClient(model="gpt-4o-mini", api_key="Your API key"),
        description="Generate a report based on the search and stock analysis results",
        system_message="You are a helpful assistant that can generate a comprehensive report on a given topic based on search and stock analysis. Don't need to add disclaimer When you done with generating the report, reply with TERMINATE.",
    )
    team = RoundRobinGroupChat([stock_analysis_agent, report_agent])
    result = await team.run(text, termination_condition=StopMessageTermination())
    print(type(result))
    return str(result)

# @app.route('/get_stock_trend', methods=['POST'])
# def get_stock_trend():
#     data = request.get_json()
#     ticker = data.get('ticker')

#     if not ticker:
#         return jsonify({'error': 'Stock ticker is required'}), 400

#     # Generate the stock trend plot
#     # (Replace this with actual stock data fetching and plotting logic)
#     plt.figure()
#     plt.title(f"Stock Trend for {ticker}")
#     plt.plot([1, 2, 3], [10, 20, 15])  # Dummy data
#     plt.xlabel("Time")
#     plt.ylabel("Price")
#     plot_path = "static/stock_trend.png"
#     plt.savefig(plot_path)
#     plt.close()

#     return jsonify({'url': f'/{plot_path}'})

# @app.route('/static/<path:filename>')
# def serve_static_file(filename):
#     return send_file(filename)

if __name__ == '__main__':
    # Ensure the static directory exists
    if not os.path.exists('static'):
        os.makedirs('static')
    app.run(debug=True)