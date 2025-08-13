# -------------------------
# BARTENDER BOT
# -------------------------

"""
Bartender bot: recommends drink recipes and lists store items to buy.
Answers only questions related to drinks, the store, and its role.
"""

# -------------------------

import getpass
import os
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, MessagesState, StateGraph
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chat_models import init_chat_model
from langchain_core.runnables.config import RunnableConfig
from dotenv import load_dotenv
load_dotenv()

# -------------------------

model = init_chat_model("gpt-4o-mini", model_provider="openai")

# -------------------------

prompt_template = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a bartender working in a liquor store. You help reccommend  \
            drink recipes and tell the customer which items to buy in your store \
            to create it based on their query. Answer ONLY questions related to  \
            drinking, the store, and your role to the best of your ability. If   \
            asked a question unrelated to your job, kindly remind the customer   \
            that you are unable to answer",
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

# -------------------------

# Define a new graph
workflow = StateGraph(state_schema=MessagesState)


# Define the function that calls the model
def call_model(state: MessagesState):
    chain_input = {"messages": state["messages"]}
    prompt_msgs = prompt_template.invoke(chain_input)
    response = model.invoke(prompt_msgs)  # actually use the prompt
    return {"messages": [response]}       # return a list



# Define the (single) node in the graph
workflow.add_node("model", call_model)
workflow.add_edge(START, "model")

# Add memory
memory = MemorySaver()
app = workflow.compile(checkpointer=memory)

# -------------------------

config: RunnableConfig = {"configurable": {"thread_id": os.getenv("THREAD_ID", "demo-thread-1")}}


# -------------------------

if __name__ == "__main__":
    config = {"configurable": {"thread_id": os.getenv("THREAD_ID", "demo-thread-1")}}
    query = "Hi! I'm Bob. Can you suggest a gin sour and what I should buy?"
    output = app.invoke({"messages": [HumanMessage(query)]}, config)
    output["messages"][-1].pretty_print()
