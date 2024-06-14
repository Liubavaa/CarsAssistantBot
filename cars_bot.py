import argparse
import pandas as pd
import os
import openai
import nest_asyncio
import deepl
import telebot
from llama_index.experimental.query_engine import PandasQueryEngine
from llama_index.agent.openai import OpenAIAgent
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.llms.openai import OpenAI
from llama_index.core.callbacks import CallbackManager, LlamaDebugHandler
from llama_index.core import Settings

# Apply asynchronous loops patch
nest_asyncio.apply()

# Get OpenAI API key
# os.environ['OPENAI_API_KEY'] = 'sk-proj-WEKTBUgM38191zybZGxHT3BlbkFJSzNFS0t7YOGCdsvHxe0u'
openai.api_key = os.environ["OPENAI_API_KEY"]


def debug():
    """
    Can set environment variables for LangChain and initialize a debug handler for tracing sub-questions.
    """
    os.environ['LANGCHAIN_ENDPOINT'] = 'https://api.smith.langchain.com'
    os.environ['LANGCHAIN_TRACING_V2'] = 'true'
    os.environ['LANGCHAIN_API_KEY'] = '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'
    os.environ['LANGCHAIN_PROJECT'] = 'llama-index-tester'

    # Using the LlamaDebugHandler to print the trace of the sub questions
    # captured by the SUB_QUESTION callback event type
    llama_debug = LlamaDebugHandler(print_trace_on_end=True)
    callback_manager = CallbackManager([llama_debug])
    Settings.callback_manager = callback_manager


def read_data():
    """
    Read and process the car availability data.
    """
    # Load the data
    file_path = 'merged_data_test_task.xlsx'
    base_df = pd.read_excel(file_path)

    # Select relevant columns
    columns_of_interest = ['Brand', 'Model', 'if 1-3 days\\ euro per day', 'Locations', 'Contact person']
    df = base_df[columns_of_interest]
    df.columns = ['brand', 'model', 'price_per_day', 'location', 'contact_person']

    # Handle some possible errors
    df.loc[:, 'brand'] = df['brand'].str.lower()
    df.loc[:, 'model'] = df['model'].str.lower()
    df.loc[:, 'location'] = df['location'].str.lower()
    df.loc[:, 'contact_person'] = df['contact_person'].str.lower()
    df = df.fillna("")

    return df


def create_engines(df):
    """
    Create query engines for handling natural language queries on the car DataFrame.
    """

    # Previous useful instruction
    #
    # instruction_str = """\ Consider that brand and models can be written in
    # different latter cases. Model can contain information about body style. Model and location could be more
    # specific in dataframe, than they are specified in query."""

    # Base engine for performing simple queries related only to dataframe
    query_engine = PandasQueryEngine(df=df, llm=OpenAI(model="gpt-4"))

    # Instruction will handle complex query which require knowledge of general information besides info in cars df
    instruction_str = """\
    "1. Create new query by replacing parts of query, which require general knowledge, with more specific car brands,
    models, body styles, prices, locations or contact persons. Result query should be easy to answer using dataframe
    with available car brand, model, price, location and contact person. Consider that model and location could
    be more specific in dataframe, than they are specified in query.\n"
    "2. Convert the query to executable Python code using Pandas.\n"
    "3. The final line of code should be a Python expression that can be called with the `eval()` function.\n"
    "4. The code should represent a solution to the query.\n"
    "5. PRINT ONLY THE EXPRESSION.\n"
    "6. Do not quote the expression.\n"
    """

    complex_query_engine = PandasQueryEngine(df=df, instruction_str=instruction_str, llm=OpenAI(model="gpt-4"))
    return [query_engine, complex_query_engine]


def query(query_engine, query_text):
    """For testing. Execute a query on engine."""
    response = query_engine.query(query_text, )
    print(str(response))


# Example queries
# query(query_engine, "List me Audi RS6 in Spain.")
# query(query_engine, "List me Audi RS6 in Spain.")
# query(query_engine, "Give me variants of Audi RS6 in Italy.")
# query(query_engine, "Give me a list of 3 cars with the lowest price in both locations.")
# query(query_engine, "Offer me a car from a company founded in 1939 in Italy")
# query(query_engine, "Find me cars for an off-road weekend in Spain.")


def create_chat_agent(query_engines):
    """
    Create a chat agent with specified query engines.
    """
    # Create tools list based on engines
    tools = [
        QueryEngineTool(
            query_engine=query_engines[0],
            metadata=ToolMetadata(
                name="cars_df_engine",
                description="useful for getting information about available cars, based on brand, model, price, "
                            "location, contact person",
            ),
        ),

        QueryEngineTool(
            query_engine=query_engines[1],
            metadata=ToolMetadata(
                name="updated_cars_df_engine",
                description="useful when query also requires general knowledge in order to specify information about "
                            "available cars brand, model, price, location, contact person",
            ),
        ),
    ]

    agent = OpenAIAgent.from_tools(tools, llm=OpenAI(model="gpt-4"))
    return agent


def get_translated_response(agent, query):
    """
    Return a translated response from the agent in the detected language.
    """
    auth_key = "64b94daa-8efe-41fd-b996-36cea87bb55f:fx"
    translator = deepl.Translator(auth_key)

    translated = translator.translate_text(
        query.encode('utf-8', errors='replace').decode('utf-8'), target_lang="EN-US"
    )
    if translated.detected_source_lang != "EN":
        response = agent.chat(translated.text)
        response = translator.translate_text(str(response), target_lang=translated.detected_source_lang)
    else:
        response = agent.chat(query)

    return response


def start_cli_chat(agent):
    """
    Start an interactive chat session with the agent in the console.
    """
    while True:
        text_input = input("User: ")
        if text_input == "/exit":
            break
        elif text_input == "/clear":
            agent.reset()
        else:
            response = get_translated_response(agent, text_input)
            print(f"Assistant: {response}")


def start_telegram_bot(agent):
    """
    Allow to start an interactive chat session with the telegram bot.
    """
    bot = telebot.TeleBot("7229319197:AAH4nAcGlmSWjC2ojvOC91TNXUrjnyeiSOU")

    @bot.message_handler(commands=['clear'])
    def clear(message):
        """Reset chat agent."""
        agent.reset()
        bot.reply_to(message, "Chat's cleared.")


    @bot.message_handler(func=lambda msg: True)
    def chat(message):
        """Use agent to answer user questions about cars."""
        text_input = message.text
        response = get_translated_response(agent, text_input)
        bot.reply_to(message, response)

    bot.infinity_polling()


def parse_arguments():
    parser = argparse.ArgumentParser(description="Choose the chat mode to run.")
    parser.add_argument('mode', choices=['cli', 'telegram'], help="Mode to run the chat: 'cli' or 'telegram'")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()

    df = read_data()
    query_engines = create_engines(df)
    agent = create_chat_agent(query_engines)

    if args.mode == 'cli':
        start_cli_chat(agent)
    elif args.mode == 'telegram':
        start_telegram_bot(agent)
