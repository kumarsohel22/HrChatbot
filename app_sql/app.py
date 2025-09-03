from langchain_community.utilities import SQLDatabase
import getpass
from langchain_ollama import OllamaLLM
import os
from dotenv import load_dotenv
from typing_extensions import TypedDict
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from typing_extensions import Annotated
from typing import TypedDict
from langchain_community.tools.sql_database.tool import QuerySQLDatabaseTool
from langgraph.graph import START, StateGraph

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import threading
import uvicorn

load_dotenv()

os.environ["LANGCHAIN_TRACING_V2"]="true"
os.environ["LANGCHAIN_API_KEY"]=os.getenv("LANGCHAIN_API_KEY")
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

connection_string = (
    "mssql+pyodbc:///?odbc_connect="
    "Driver={ODBC Driver 17 for SQL Server};"
    "Server=LAPTOP-E5BJ5PM9;"
    "Database=HRDatabase;"
    "Trusted_Connection=yes;"
)

# -----------------------------
# 2. Load SQLDatabase
# -----------------------------
db = SQLDatabase.from_uri(connection_string)
db.run("SELECT * FROM Employees;")

app = FastAPI()

class QuestionRequest(BaseModel):
    question: str

class AnswerResponse(BaseModel):
    answer: str
    query: str
    result: str


class State(TypedDict):
    question: str
    query: str
    result: str
    answer: str

llm = init_chat_model("llama-3.3-70b-versatile", model_provider="groq")

system_message = """
Given an input question, create a syntactically correct {dialect} query to
run to help find the answer. Unless the user specifies in his question a
specific number of examples they wish to obtain. You can order the results by a relevant column to
return the most interesting examples in the database.

Never query for all the columns from a specific table, only ask for a the
few relevant columns given the question.

Pay attention to use only the column names that you can see in the schema
description. Be careful to not query for columns that do not exist. Also,
pay attention to which column is in which table.

Only use the following tables:
{table_info}
"""

user_prompt = "Question: {input}"

query_prompt_template = ChatPromptTemplate(
    [("system", system_message), ("user", user_prompt)]
)

for message in query_prompt_template.messages:
    message.pretty_print()

class QueryOutput(TypedDict):
    """Generated SQL query."""

    query: Annotated[str, ..., "Syntactically valid SQL query."]


def write_query(state: State):
    """Generate SQL query to fetch information."""
    prompt = query_prompt_template.invoke(
        {
            "dialect": db.dialect,
            "top_k": 10,
            "table_info": db.get_table_info(),
            "input": state["question"],
        }
    )
    structured_llm = llm.with_structured_output(QueryOutput)
    result = structured_llm.invoke(prompt)
    print("Generated SQL Query:", result)
    return {"query": result["query"]}

def execute_query(state: State):
    """Execute SQL query."""
    execute_query_tool = QuerySQLDatabaseTool(db=db)
    return {"result": execute_query_tool.invoke(state["query"])}

def generate_answer(state: State):
    """Answer question using retrieved information as context."""
    prompt = (
        "Given the following user question, corresponding SQL query, "
        "and SQL result, answer the user question.\n\n"
        f"question: {state['question']}\n"
        f"SQL Query: {state['query']}\n"
        f"SQL Result: {state['result']}"
    )
    response = llm.invoke(prompt)
    return {"answer": response.content}

graph_builder = StateGraph(State).add_sequence(
    [write_query, execute_query, generate_answer]
)
graph_builder.add_edge(START, "write_query")
graph = graph_builder.compile()

# for step in graph.stream(
#     {"question": "How many employees are unavailable?"}, stream_mode="updates"
# ):
#     print(step)

# @app.post("/ask", response_model=AnswerResponse)
# async def ask_question(request: QuestionRequest):
#     try:
#         # state = {"question": request.question}
#         state = {
#             "question": request.question,
#             "query": "",
#             "result": "",
#             "answer": ""
#         }

#         final_state = {}
#         for step in graph.stream({"question": request.question}, stream_mode="updates"):
#             final_state.update(step)
#             print("step", step)
#         return final_state
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))

@app.post("/ask", response_model=AnswerResponse)
async def ask_question(request: QuestionRequest):
    try:
        final_state = {"answer": "", "query": "", "result": ""}

        # Run the graph stream
        for step in graph.stream({"question": request.question}, stream_mode="updates"):
            print("step", step)
            if "write_query" in step:
                final_state["query"] = step["write_query"]["query"]
            if "execute_query" in step:
                final_state["result"] = str(step["execute_query"]["result"])
            if "generate_answer" in step:
                final_state["answer"] = step["generate_answer"]["answer"]

        # Ensure strings for FastAPI validation
        return {
            "answer": str(final_state["answer"]),
            "query": str(final_state["query"]),
            "result": str(final_state["result"]),
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)

