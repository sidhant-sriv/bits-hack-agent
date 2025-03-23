from langchain.schema import Document
from tools import (
    retriever,
    rag_chain,
    retrieval_grader,
    web_search_tool,
    question_router,
    intent_classifier,
    hallucination_grader,
    answer_grader,
)


def retrieve(state):
    """
    Retrieve documents from vectorstore

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, documents, that contains retrieved documents
    """
    question = state["question"]
    # Initialize documents as empty list
    documents = []
    # Retrieval
    if retriever is not None:
        documents = retriever.invoke(question)

    return {"documents": documents, "question": question}


#
def generate(state):
    """
    Generate answer using RAG on retrieved documents

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, generation, that contains LLM generation
    """
    question = state["question"]
    documents = state["documents"]

    # RAG generation
    generation = rag_chain.invoke({"context": documents, "question": question})
    return {"documents": documents, "question": question, "generation": generation}


#
def grade_documents(state):
    """
    Determines whether the retrieved documents are relevant to the question
    If any document is not relevant, we will set a flag to run web search

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Filtered out irrelevant documents and updated web_search state
    """

    print("---CHECK DOCUMENT RELEVANCE TO QUESTION---")
    question = state["question"]
    documents = state["documents"]

    # Score each doc
    filtered_docs = []
    web_search = "No"
    for d in documents:
        score = retrieval_grader.invoke(
            {"question": question, "document": d.page_content}
        )
        grade = score["score"]
        # Document relevant
        if grade.lower() == "yes":
            filtered_docs.append(d)
        # Document not relevant
        else:
            web_search = "Yes"
            continue
    return {"documents": filtered_docs, "question": question, "web_search": web_search}


#
def web_search(state):
    """
    Web search based based on the question

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Appended web results to documents
    """

    question = state["question"]
    # Initialize documents as empty list if it doesn't exist
    documents = state.get("documents", [])

    # Web search
    docs = web_search_tool.invoke({"query": question})
    web_results = "\n".join([d["content"] for d in docs])
    web_results = Document(page_content=web_results)
    if documents:
        documents.append(web_results)
    else:
        documents = [web_results]
    return {"documents": documents, "question": question}


def route_question(state):
    """
    Route question to web search or RAG.

    Args:
        state (dict): The current graph state

    Returns:
        str: Next node to call
    """

    question = state["question"]
    print(question)
    source = question_router.invoke({"question": question})
    print(source)
    print(source["datasource"])
    if source["datasource"] == "web_search":
        return "websearch"
    elif source["datasource"] == "vectorstore":
        return "vectorstore"


def route_intent(state):
    question = state["question"]
    classification = intent_classifier.invoke({"question": question})
    return classification


# Conditional edges
def decide_to_generate(state):
    """
    Determines whether to generate an answer, or add web search

    Args:
        state (dict): The current graph state

    Returns:
        str: Binary decision for next node to call
    """

    question = state["question"]
    web_search = state["web_search"]
    filtered_documents = state["documents"]

    if web_search == "Yes":
        # All documents have been filtered check_relevance
        # We will re-generate a new query

        return "websearch"
    else:
        # We have relevant documents, so generate answer

        return "generate"


def grade_generation_v_documents_and_question(state):
    """
    Determines whether the generation is grounded in the document and answers question.

    Args:
        state (dict): The current graph state

    Returns:
        str: Decision for next node to call
    """

    question = state["question"]
    documents = state["documents"]
    generation = state["generation"]

    score = hallucination_grader.invoke(
        {"documents": documents, "generation": generation}
    )
    grade = score["score"]

    # Check hallucination
    if grade == "yes":
        score = answer_grader.invoke({"question": question, "generation": generation})
        grade = score["score"]
        if grade == "yes":
            return "useful"
        else:
            return "not useful"
    else:
        return "not supported"


def entry_data(state):
    """
    This node doesn't modify the state - it simply passes it through without changes.
    This can be useful for debugging or as a placeholder in the workflow.

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): The unchanged state
    """
    print(
        "This goes as an entry for parallel action of getting user data and form data"
    )
    return state


def user_data_sql(state):
    # Get user data
    user_data = {}
    return {"user_data": user_data}


def get_form_struct(state):
    # form data from RAG

    form_struct = {}
    return {"form_struct": form_struct}


def merge_node(state):
    # Takes keys from user_data and adds it to form struct
    new_form_stuct = {"name": "sidhant"}
    return {"form_struct": new_form_stuct}
