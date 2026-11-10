import os
import subprocess
import uuid
from typing import TypedDict, Annotated, Literal
from dotenv import load_dotenv

from pydantic import BaseModel, Field

from langchain_ollama import ChatOllama
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.documents import Document
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import interrupt, Command

load_dotenv()

llm = ChatOllama(model='llama3.2:1b')
embedding = OllamaEmbeddings(model='all-minilm')

KNOWLEDGE = [
    'The Greenhouse Gas Protocol categorizes corporate emissions into Scope 1 (direct), Scope 2 (indirect from purchased energy), and Scope 3 (value chain).',
    'The Science Based Targets initiative validates corporate decarbonisation goals to ensure alignment with the Paris Agreement target of limiting global warming to 1.5°C.',
    'Net-zero decarbonisation protocols require organizations to reduce absolute greenhouse gas emissions by at least 90 percent and neutralize any residual emissions.',
    'ISO 14064 provides an international standard for quantifying, monitoring, reporting, and verifying greenhouse gas emissions and removals.',
    'Transitioning to renewable energy sources and improving industrial energy efficiency serve as primary foundational mechanisms for reducing operational emissions.'
]

vector_store = InMemoryVectorStore(embedding=embedding)
vector_store.add_documents([Document(page_content=text) for text in KNOWLEDGE])

class IntentClassifier(BaseModel):
    message_intent: Literal['chat', 'knowledge', 'code'] = Field(
        description='The destination selected according to the routing rules.'
    )
structured_llm = llm.with_structured_output(IntentClassifier)
#Routing_rule to reduce ambiguity in meaning of knowledge, chat, and code for llama.
ROUTING_RULES = '''
Choose one destination for the user's message. Apply these rules in order.

1. code: The user asks to create, edit, fix, or delete code in this project.
2. knowledge: The user explicitly asks to look something up in their notes
   or documents, or asks a factual question about a topic in the stored
   passages. The passages cover greenhouse gas emissions scopes, science-based
   targets, net zero, ISO 14064, renewable energy, and energy efficiency.
3. chat: Use the general assistant for greetings, casual conversation,
   and other requests.
Examples:
'Hi!' -> chat
'Hi, what are Scope 1 emissions?' -> knowledge
'What do my notes say about basketball?' -> knowledge
'Can you explain basketball sponsorship?' -> chat
'Change my Python code to calculate Scope 1 emissions.' -> code
'''

class State(TypedDict):
    messages: Annotated[list, add_messages] #this line defines a list of messages in the state, and the add_messages decorator is used to specify that this field should be treated as a collection of messages. 
    message_intent: str | None #this line defines a message_intent field in the state, which can be either a string or None. 
    next_node: str | None #this line defines a next_node field in the state, which can be either a string or None.

#Classifier node
def classify_intent(state: State):
    result = structured_llm.invoke([
        {'role': 'system', 'content': ROUTING_RULES},
        {'role': 'user', 'content': state['messages'][-1].content},
    ])
    return {'message_intent': result.message_intent}

#Human Decision Node
def accept_coding(state: State):
    user_prompt = state["messages"][-1].content

    decision = interrupt(
        f"About to run Aider with this request:\n\n{user_prompt}\n\n"
        "Type y to approve, n to cancel, or enter a revised request to run."
    )

    text = str(decision).strip()
    choice = text.lower()

    if not text or choice in {"n", "no", "deny", "cancel"}:
        return {"messages": [{"role": "assistant", "content": "Coding request cancelled.",}], 
                "next_node": "denied",}

    if choice in {"y", "yes", "approve", "ok"}:
        return {"next_node": "coding_agent"}
    
    return {
        'messages': [{'role': 'user', 'content': text}],
        'next_node': 'accept_coding',
    }

#Chat node
def prompt_llm_chat(state: State):
    messages = [{'role': 'system', 'content': 'You are a helpful chatbot. Be straightforward.'}] + state['messages']

    response = llm.invoke(messages)

    return {'messages': [{'role': 'assistant', 'content': response.content}]}

#RAG node (Message designed specifically to gives Llama only the retrieved passages and the current question, so earlier assistant replies can’t introduce extra information.)
def prompt_llm_rag(state: State):

    query = state['messages'][-1].content

    documents = vector_store.similarity_search(query, k=3)

    context = '\n'.join(f'-{doc.page_content}' for doc in documents)

    messages = [
        {
            "role": "system",
            "content": (
                "Answer the question using only the passages below.\n"
                "If a passage explicitly answers the whole question, copy the "
                "shortest excerpt that fully answers it. Return only those words.\n"
                "If any requested detail is missing, reply exactly: I don't know.\n"
                "A passage about the same topic is sufficient only if it gives "
                "the specific information requested. For example, a definition "
                "of Scope 1 emissions cannot answer a question asking for a "
                "particular company's emissions figure.\n"
                "Do not add examples, explanations, guesses, or introductory text.\n\n"
                f"Passages:\n{context}"
            ),
        },
        {"role": "user", "content": query},
    ]

    response = llm.invoke(messages)

    return {'messages': [{'role': 'assistant', 'content': response.content}]}

#Code node
def prompt_llm_code(state: State):

    user_prompt = state['messages'][-1].content

    #get the directory of the current script file (main.py) 
    workspace = os.path.join(os.path.dirname(os.path.abspath(__file__))) 
    print(f"workspace: {workspace}")

    result = subprocess.run(
        [
            "aider",
            "--model", "ollama_chat/qwen2.5-coder:1.5b",
            "--no-git",
            "--yes-always",
            "--map-tokens", "0",
            "--message", user_prompt,
        ],
        cwd=workspace,
        capture_output=True,
        text=True,
    )

    output = result.stdout.strip() or result.stderr.strip()

    return {'messages': [{'role': 'assistant', 'content': output}]}

#prepare coding request node
def prepare_coding_request(state: State):
    messages = [{
            'role': 'system',
            'content': (
                'Rewrite the latest user coding request into a clear instruction for Aider. '
                'Use the conversation history as context. Only output the instruction, no explanation.'
            ),
            },
            ] + state['messages']

    response = llm.invoke(messages)

    return {'messages': [{'role': 'assistant', 'content': response.content}]}

#After defining the nodes, we can now build the graph by adding the nodes and edges to the StateGraph.
graph_builder = StateGraph(State)

graph_builder.add_node('classifier',classify_intent)
graph_builder.add_node('chat_agent',prompt_llm_chat)
graph_builder.add_node('rag_agent',prompt_llm_rag)
graph_builder.add_node('coding_agent',prompt_llm_code)
graph_builder.add_node('prepare_coding_request', prepare_coding_request)
graph_builder.add_node('accept_coding', accept_coding)


graph_builder.add_edge(START, 'classifier')
graph_builder.add_edge('prepare_coding_request', 'accept_coding')
graph_builder.add_conditional_edges('accept_coding', lambda state: state['next_node'], 
    {
        'denied': END,
        'coding_agent': 'coding_agent',
        'accept_coding': 'prepare_coding_request',
    },
)
graph_builder.add_conditional_edges('classifier', lambda state: state['message_intent'], 
    {
        'chat': 'chat_agent', 
        'knowledge': 'rag_agent', 
        'code': 'prepare_coding_request'
    })
graph_builder.add_edge('chat_agent', END)
graph_builder.add_edge('rag_agent', END)
graph_builder.add_edge('coding_agent', END)

checkpointer = InMemorySaver()
graph = graph_builder.compile(checkpointer=checkpointer)

graph.get_graph().draw_mermaid_png(output_file_path='graph.png')

config = {'configurable': {'thread_id': str(uuid.uuid4())}}

while True:
    user_message = input('Enter message: ')
    result = graph.invoke({'messages': [{'role': 'user', 'content': user_message}]}, config=config)

    while '__interrupt__' in result:
        prompt = result['__interrupt__'][0].value #this line extracts the first interrupt message from the result dictionary and assigns it to the prompt variable. 
        decision = input(f'{prompt}\n>') #this line prompts the user for input, displaying the interrupt message and waiting for the user's decision.
        result = graph.invoke(Command(resume=decision), config=config) #this line resumes the graph execution with the user's decision

    print('Classified intent:', result['message_intent'])
    print('Reply:', result['messages'][-1].content)


