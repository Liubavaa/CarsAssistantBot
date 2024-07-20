# Cars Assistant Bot

Provide CLI and Telegram versions of AI-Bot based on ChatGPT-4. The bot processed the included dataset (```merged_data_test_task.xlsx```). The dataset contains information about available cars in different countries. As a result, bot can assist to find best option among all cars.

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
   Price per day: 865
   Contact person: Anton Kostin Lydia

3. Location: Malaga - Madrid, Spain
   Price per day: 865
   Contact person: Matteo / Superauto +34607068589 / 34 609 21 9...

4. Location: Madrid - Caribi, Spain
   Price per day: 865
   Contact person: Emilio

5. Location: Madrid - Malaga, Spain
   Price per day: 865
   Contact person: Not provided

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
   Price per day: 1650
   Location: Ibiza - Marbella - Mallorca - Madrid - Barcelona...
   Contact person: Munich Cars Quality & Beringcars 39 351 770 9362

3. Brand: Ferrari
   Model: 488 Spider
   Price per day: 1750
   Location: Ibiza - Marbella - Mallorca - Madrid - Barcelona...
   Contact person: Munich Cars Quality & Beringcars 39 351 770 9362

Please note that there are many more options available. Let me know if you need more information or if you have a specific model in mind.

##
