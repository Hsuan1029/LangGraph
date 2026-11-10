import uuid
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langgraph.graph import MessagesState, StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver

load_dotenv()

# Specify the model name and model_provider separately
llm = init_chat_model('llama3.2', model_provider='ollama')

# Define a function to prompt the LLM with the current messages
def prompt_llm(state: MessagesState):
    response = llm.invoke(state['messages']) 
    return {'messages': [response]}

graph_builder = StateGraph(MessagesState)

graph_builder.add_node(prompt_llm)
graph_builder.add_edge(START, 'prompt_llm')
graph_builder.add_edge('prompt_llm', END)


#generate a graph with the defined nodes and edges, and use InMemorySaver to save the state in memory.
graph = graph_builder.compile(checkpointer=InMemorySaver()) 

#define a unique thread_id for each conversation to maintain context.
config = {'configurable': {'thread_id': str(uuid.uuid4())}} 

while True:
    user_message = input("Enter message: ")
    print(graph.invoke({'messages': [{'role': 'user', 'content': user_message}]}, config=config))