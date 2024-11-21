from flask import Flask, request, jsonify, send_file
import matplotlib.pyplot as plt
import os

app = Flask(__name__)

@app.route('/')
def index():
    return send_file('index.html')  # Ensure index.html is in the same directory

@app.route('/get_stock_trend', methods=['POST'])
def get_stock_trend():
    data = request.get_json()
    ticker = data.get('ticker')

    if not ticker:
        return jsonify({'error': 'Stock ticker is required'}), 400

    # Generate the stock trend plot
    # (Replace this with actual stock data fetching and plotting logic)
    plt.figure()
    plt.title(f"Stock Trend for {ticker}")
    plt.plot([1, 2, 3], [10, 20, 15])  # Dummy data
    plt.xlabel("Time")
    plt.ylabel("Price")
    plot_path = "static/stock_trend.png"
    plt.savefig(plot_path)
    plt.close()

    return jsonify({'url': f'/{plot_path}'})

@app.route('/static/<path:filename>')
def serve_static_file(filename):
    return send_file(filename)

if __name__ == '__main__':
    # Ensure the static directory exists
    if not os.path.exists('static'):
        os.makedirs('static')
    app.run(debug=True)