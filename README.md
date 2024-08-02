# Cars Assistant Bot

This project implements an interactive chat system using OpenAI's language models, designed to provide information about car availability based on a dataset. The system supports two modes of interaction: a command-line interface (CLI) and a Telegram bot. It leverages natural language processing to answer user queries about cars, including brand, model, price, location, and contact person. Additionally, it can handle more complex queries requiring general knowledge.


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

### Example of usage
##
- Me:
List me Audi RS6 in Spain.
  
- AI:
Here are some options for Audi RS6 in Spain:

1. Location: Marbella, Spain
   Price per day: 865
   Contact person: Mr. Good1

2. Location: Barcelona, Spain
   ...

##
- You:
Offer me a car from a company founded in 1939 in Italy

- AI:
Here are some options for cars from a company founded in 1939 in Italy:

1. Brand: Ferrari
   Model: Portofino M
   Price per day: 1150
   Location: Ibiza - Marbella - Mallorca - Madrid - Barcelona...
   Contact person: Munich Cars Quality & Beringcars 39 351 770 9362

2. Brand: Ferrari
   Model: F8
   ...
##
