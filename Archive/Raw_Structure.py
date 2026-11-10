import uuid
from typing import TypedDict, Annotated, Literal
from dotenv import load_dotenv

from pydantic import BaseModel, Field

from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import InMemorySaver

load_dotenv()

llm = ChatOllama(model="llama3.2:1b")


class IntentClassifier(BaseModel):
    message_intent: Literal["chat", "knowledge", "code"] = Field(...,
        description="Classify whether the user wants to just chat, ask for knowledge or change code in the project."
    )

class State(TypedDict):
    messages: Annotated[list, add_messages] #this line defines a list of messages in the state, and the add_messages decorator is used to specify that this field should be treated as a collection of messages. 
    message_intent: str | None #this line defines a message_intent field in the state, which can be either a string or None. 

#Classifier node
def classify_intent(state: State):
    structured_llm = llm.with_structured_output(IntentClassifier)
    result = structured_llm.invoke([
        {'role': 'system', 'content': 'Determine / classify whether the user wants to chat ("chat"), retrieve knowledge from their notes ("knowledge"), or change code in the project ("code").'},
        {'role': 'user', 'content': state['messages'][-1].content}
    ])
    return {"message_intent": result.message_intent}

#Chat node
def prompt_llm_chat(state: State):

    messages = [{'role': 'system', 'content': 'You are a talkative chatbot for fun. Be nice'}] + state['messages']

    response = llm.invoke(messages)

    return {'messages': [{'role': 'assistant', 'content': response.content}]}

#RAG node (Message designed specifically to gives Llama only the retrieved passages and the current question, so earlier assistant replies can’t introduce extra information.)
def prompt_llm_rag(state: State):

    messages = [{'role': 'system', 'content': 'Always say "I am the RAG agent.", no exception'}] + state['messages']

    response = llm.invoke(messages)

    return {'messages': [{'role': 'assistant', 'content': response.content}]}

#Code node
def prompt_llm_code(state: State):

    messages = [{'role': 'system', 'content': 'No matter what the user says, always say "I am the Code agent."'}] + state['messages']

    response = llm.invoke(messages)

    return {'messages': [{'role': 'assistant', 'content': response.content}]}

#After definining the nodes, we can now build the graph by adding the nodes and edges to the StateGraph.
graph_builder = StateGraph(State)

graph_builder.add_node('classifier',classify_intent)
graph_builder.add_node('chat_agent',prompt_llm_chat)
graph_builder.add_node('rag_agent',prompt_llm_rag)
graph_builder.add_node('coding_agent',prompt_llm_code)

graph_builder.add_edge(START, 'classifier')
graph_builder.add_conditional_edges('classifier', lambda state: state['message_intent'], {'chat': 'chat_agent', 'knowledge': 'rag_agent', 'code': 'coding_agent'})
#lambda function extract the message_intent from the state. If the intent is chat, the graph will transition to the chat_agent node..etc

graph_builder.add_edge('chat_agent', END)
graph_builder.add_edge('rag_agent', END)
graph_builder.add_edge('coding_agent', END)

checkpointer = InMemorySaver()
graph = graph_builder.compile(checkpointer=checkpointer)

config = {'configurable': {'thread_id': str(uuid.uuid4())}}

while True:
    user_message = input("Enter message: ")
    result = graph.invoke({'messages': [{'role': 'user', 'content': user_message}]}, config=config)

    print("Classified intent:", result["message_intent"])
    print("Reply:", result["messages"][-1].content)


