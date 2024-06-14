# Cars Assistant Bot

### Installing

- clone repository

- set up a virtual environment (optional but recommended)
```
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
```

- install the required packages
```
pip install -r requirements.txt
```

- set openai api key
```
export OPENAI_API_KEY='your_openai_api_key'
```

##
### Executing program

##
#### CLI Bot

In order to start CLI bot run:
```
python cars_bot.py cli
```
Then wait for the "User:" line to appear in your terminal and try asking questions.

##
#### Telegram Bot

In order to start Telegram bot run:
```
python cars_bot.py telegram
```
Then find @available_cars_AI_bot in telegram and try asking questions.
