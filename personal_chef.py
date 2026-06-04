from dotenv import load_dotenv

load_dotenv()

from langchain.tools import tool
from typing import Dict, Any
from tavily import TavilyClient

from langchain.agents import create_agent
from langchain.checkpoint.memory import InMemorySaver


from langchain.messages import HumanMessages

tavily_client=TavilyClient()

@tool
def web_search(query: str)->Dict[str, Any]:
    """Search the web for information"""
    return tavily_client.search(query)

system_prompt="""
    You are profession personal chef user will provide you with the ingrideants he has and you will seacrh the web and return what ever reciepe possible and also return the instruction to mak eit if requeasted
"""

agent=create_agent(
    model="gpt-5-nano",
    tools=[web_search],
    system_prompt=system_prompt,
    checkpointer=InMemorySaver()
)

config={"configurable":{"thread_id":"1"}}

response=agent.invoke(
    {"messages":[HumanMessages(content="")]},
    config
)

print(response["messages"][-1].content)