README for the CS410 Project Repository.

Maintainers:
Dmitry Smirnov (dmitrys3), Charles Yu (cyu66), Haosen Yao (haoseny2), Lambodar Panigrahy (lp18)

Link to video presentation: https://mediaspace.illinois.edu/media/t/1_lh8wm41i

How to Run:
You will need a gnews api key (free), and a gpt agent api key (minimum of $5) in the root level .env . 
https://gnews.io/
https://platform.openai.com/docs/api-reference/introduction

Python >= 3.10 is necessary
All required libraries are contained in the repo requirements.txt file. 
Command to run the agent: python app.py
Command to test sentiment: in the data_collection directory run python article_sentiment.py

Agent accepts commands such as 'Financial analysis of <Ticket>' or 'Report on <Ticker>'

The web widget will be available on http://127.0.0.1:5000/ once app.py is run. 
