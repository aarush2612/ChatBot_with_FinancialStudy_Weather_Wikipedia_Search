from langchain_google_genai import ChatGoogleGenerativeAI
import chainlit as cl
from langchain.agents import ZeroShotAgent, Tool, AgentExecutor
from langchain.tools import BaseTool
from langchain import LLMChain
from typing import Optional
from langchain.utilities import GoogleSearchAPIWrapper
from langchain import LLMMathChain, SerpAPIWrapper
from langchain.agents import initialize_agent, Tool
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferWindowMemory
import yfinance as yf
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain.callbacks.base import BaseCallbackHandler
import plotly.graph_objects as go
from distutils.util import strtobool
from langchain_groq import ChatGroq



@cl.on_chat_start
def start():
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro-001")
    llm1 = ChatGroq(model="llama3-8b-8192")
    search = SerpAPIWrapper()
    llm_math_chain = LLMMathChain(llm=llm, verbose=True)
    wikipedia = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())

    class GraphTool(BaseTool):
        name = "FinanceGraph"
        description = "best to plot charts"

        def _run(self, query: str) -> str:
            tk = yf.Ticker("AAPL")

            company_info = {
                "info": tk.info
            }
            return "Here's a chart [chart of 'ticker']"

    class CustomTool(BaseTool):
        name = "CPFinance"
        description = "Provides financial data such as current price, market cap, and other financial information"

        def _run(self, query: str) -> str:
            tk = yf.Ticker(llm.invoke(f"Reply with only ticker for the company, only one word, using, Query(add .NS at the end if it's an indian stock): {query}").content.strip())

            company_info = {
                "info": tk.info
            }
            return str(company_info)

    tools = [
        GraphTool(),

        Tool(
            name="Search",
            func=search.run,
            description="Useful for answering questions about current events or questions that require logical analysis. Ask targeted questions for best results."
        ),
        Tool(
            name="Calculator",
            func=llm_math_chain.invoke,
            description="Handles mathematical queries and calculations."
        ),
        CustomTool(),
        Tool(
            name="Wikipedia",
            func=wikipedia.run,
            description="Best for answering questions about historical events or general knowledge not related to current financial data."
        )
    ]

    prefix = """You are a financial advisor and friend of the user. Always provide detailed yet concise answers. You have access to the following tools if absolutely necessary:"""
    suffix = """Begin!

    {chat_history}
    Question: {input}
    {agent_scratchpad}"""

    prompt = ZeroShotAgent.create_prompt(
        tools,
        prefix=prefix,
        suffix=suffix,
        input_variables=["input", "chat_history", "agent_scratchpad"]
    )
    memory = ConversationBufferWindowMemory(memory_key="chat_history", k=20)

    llm_chain = LLMChain(llm=llm, prompt=prompt)
    agent = ZeroShotAgent(llm_chain=llm_chain, tools=tools, verbose=True)
    agent_chain = AgentExecutor.from_agent_and_tools(agent=agent, tools=tools, verbose=True, memory=memory, handle_parsing_errors=True)
    cl.user_session.set("agent_chain", agent_chain)
    cl.user_session.set("llm", llm)
    cl.user_session.set("llm1", llm1)

@cl.on_message
async def main(message: cl.Message):
    agent = cl.user_session.get("agent_chain")
    llm = cl.user_session.get("llm")
    llm1 = cl.user_session.get("llm1")

    if isinstance(message, cl.message.Message):
        message = message.content

    cb = cl.LangchainCallbackHandler(stream_final_answer=True)

    response = await cl.make_async(agent.run)(message, callbacks=[cb])

    # Not added in tool beacause fig object is not passable currenly.
    if strtobool(llm1.invoke(f"Reply with only True or False. Is the following statement talking about drawing a visual financial chart/graph: {response}").content.strip()) or "chart" in message.lower():
        ticker = llm.invoke(f"Reply with only ticker for the company, only one word(add .NS if it's an indian stock), using, Query: {message}---AND---respones: {response}").content.strip()
        df = yf.download(ticker, period="1y")

        fig = go.Figure(
            data=[go.Scatter(x=df.index, y=df['Close'], mode='lines')],
            layout_title_text=f"{ticker} (1y)"
        )
        fig.update_xaxes(title_text='Date')
        fig.update_yaxes(title_text='Price')

        elements = [cl.Plotly(name="chart", figure=fig, display="inline")]

        await cl.Message(content=f"Here's the chart for {ticker}", elements=elements).send()
    else:
        msg = cl.Message(content="")
        for chunk in response:
            await msg.stream_token(chunk)

        await msg.send()