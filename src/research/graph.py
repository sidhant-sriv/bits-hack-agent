from fastapi import FastAPI, HTTPException, Body
from pydantic import BaseModel
from typing import Dict, List, Any, Optional
from pydantic_core import core_schema
import uvicorn
from langgraph.graph import END, StateGraph
from state import GraphState
from fastapi.middleware.cors import CORSMiddleware

# Import directly from the same file
from nodes import (
    decide_to_generate,
    generate,
    get_form_struct,
    grade_documents,
    grade_generation_v_documents_and_question,
    merge_node,
    retrieve,
    route_intent,
    route_question,
    user_data_sql,
    web_search,
    entry_data,
)



# Initialize the workflow
workflow = StateGraph(GraphState)
# Define the nodes
workflow.add_node("websearch", web_search)  # web search
workflow.add_node("retrieve", retrieve)  # retrieve
workflow.add_node("grade_documents", grade_documents)  # grade documents
workflow.add_node("generate", generate)
workflow.add_node("entry_data", entry_data)
workflow.add_node("user_data_sql", user_data_sql)
workflow.add_node("get_form_struct", get_form_struct)
workflow.add_node("merge_node", merge_node)
workflow.add_node("route_intent", route_intent)
workflow.set_conditional_entry_point(
    route_question,
    {
        "websearch": "websearch",
        "vectorstore": "retrieve",
    },
)
workflow.add_edge("retrieve", "grade_documents")
workflow.add_conditional_edges(
    "grade_documents",
    decide_to_generate,
    {
        "websearch": "websearch",
        "generate": "generate",
    },
)
workflow.add_edge("websearch", "generate")
workflow.add_conditional_edges(
    "generate",
    grade_generation_v_documents_and_question,
    {
        "not supported": "generate",
        "useful": "route_intent",
        "not useful": "websearch",
    },
)
workflow.add_conditional_edges(
    "route_intent",
    lambda x: x["intent"],
    {"question_answering": END, "command": "entry_data"},
)
workflow.add_edge("entry_data", "user_data_sql")
workflow.add_edge("entry_data", "get_form_struct")
workflow.add_edge("user_data_sql", "merge_node")
workflow.add_edge("get_form_struct", "merge_node")
workflow.add_edge("merge_node", END)
app_workflow = workflow.compile()


# Create FastAPI app
app = FastAPI(
    title="LangGraph Workflow API",
    description="API for executing a LangGraph workflow for RAG and form processing",
    version="1.0.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods
    allow_headers=["*"],  # Allow all headers
)


class QueryRequest(BaseModel):
    question: str
    user_id: int
    context: Optional[Dict[str, Any]] = None


class WorkflowResponse(BaseModel):
    generation: Optional[str] = None
    form_struct: Optional[Dict[str, Any]] = None
    user_id: Optional[int] = None


@app.post("/query", response_model=WorkflowResponse)
async def execute_query(request: QueryRequest = Body(...)):
    """
    Execute a query through the LangGraph workflow.

    This endpoint processes a question through the workflow which may:
    - Retrieve documents from a vectorstore
    - Search the web if necessary
    - Generate an answer based on the retrieved information
    - Classify intent and potentially process form data

    Required parameters:
    - question: The user's question or command
    - user_id: The unique identifier for the user

    Optional parameters:
    - context: Additional context information
    """
    try:
        # Initialize the state with the question and user_id
        initial_state = {"question": request.question, "context": request.user_id}

        # Execute the workflow
        result = await app_workflow.ainvoke(initial_state, {"recursion_limit": 10})

        # Parse the output for the response
        response = WorkflowResponse()

        # Add available fields from the result to the response
        if "generation" in result:
            response.generation = result["generation"]
        if "form_struct" in result:
            response.form_struct = result["form_struct"]
        if "user_id" in result:
            response.user_id = result["user_id"]
        else:
            response.user_id = request.user_id

        return response

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Workflow execution failed: {str(e)}"
        )


@app.get("/")
async def root():
    """Health check endpoint"""
    return {"status": "API is running", "version": "1.0.0"}


# Update user_data_sql function to utilize the user_id
def custom_user_data_sql(state):
    """
    Modified version that uses the user_id from the state
    """
    user_id = state.get("user_id")
    # Get user data based on user_id
    user_data = {"user_id": user_id}
    # Here you would normally query a database using the user_id
    return {"user_data": user_data, "user_id": user_id}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
