from langgraph.graph import END, StateGraph
from research.nodes import (
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
from state import GraphState

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
        "useful": "route_intent",  # Now this points to a valid node
        "not useful": "websearch",
    },
)

workflow.add_conditional_edges(
    "route_intent",
    lambda x: x["intent"],  # Use the output of the route_intent function
    {"question_answering": END, "command": "entry_data"},
)
workflow.add_edge("entry_data", "user_data_sql")
workflow.add_edge("entry_data", "get_form_struct")
workflow.add_edge("user_data_sql", "merge_node")
workflow.add_edge("get_form_struct", "merge_node")
workflow.add_edge("merge_node", END)

app = workflow.compile()
