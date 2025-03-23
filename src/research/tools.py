from groq import Groq
from langchain_groq import ChatGroq
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_cohere import CohereEmbeddings

from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from pydantic import SecretStr
import os
from langchain_community.tools.tavily_search import TavilySearchResults
from dotenv import load_dotenv

load_dotenv()
embeddings = CohereEmbeddings(
    model="embed-english-v3.0", client=None, async_client=None
)
groq_api_key = os.environ.get("GROQ_API_KEY")


def load_vectorstore_from_disk(persist_directory="chroma.db"):
    """
    Load a Chroma vectorstore from disk.

    Args:
        persist_directory: Path to the directory where Chroma data is stored, defaults to "chroma.db"

    Returns:
        Chroma vectorstore object
    """
    try:
        vectorstore = Chroma(
            persist_directory=persist_directory, embedding_function=embeddings
        )
        return vectorstore
    except Exception as e:
        print(f"Error loading vectorstore: {e}")
        return None


vectorstore = load_vectorstore_from_disk()

if vectorstore is not None:
    retriever = vectorstore.as_retriever(search_kwargs={"k": 7})
else:
    print("Failed to load vectorstore, retriever not initialized")
    retriever = None

router_llm = ChatGroq(
    temperature=0,
    model="llama3-8b-8192",
    api_key=SecretStr(groq_api_key) if groq_api_key else None,
)

llm = ChatGroq(
    temperature=0,
    model="llama-3.3-70b-specdec",
    api_key=SecretStr(groq_api_key) if groq_api_key else None,
)

prompt_question = PromptTemplate(
    template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|> You are an expert at routing a 
    user question to a vectorstore or web search. Use the vectorstore for questions on copyright i.e Introduction to the Copyright Act, 1957, The Copyright (Amendment) Act, 2012, Reasons for Amendments, Digital Protection, Internet Service Provider (ISP) Liability, Statutory Licenses, Royalty Rights, Performer Rights, Copyright Societies, Exceptions for the Physically Disabled. You do not need to be stringent with the keywords 
    in the question related to these topics. Otherwise, use web-search. Give a binary choice 'web_search' 
    or 'vectorstore' based on the question. Return the a JSON with a single key 'datasource' and 
    no premable or explaination. Question to route: {question} <|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
    input_variables=["question"],
)

question_router = prompt_question | router_llm | JsonOutputParser()
prompt = PromptTemplate(
    template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|> You are an assistant for question-answering tasks. 
    Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. 
    Try to give as much information as possible.
    c <|eot_id|><|start_header_id|>user<|end_header_id|>
    Question: {question} 
    Context: {context} 
    Answer: <|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
    input_variables=["question", "document"],
)


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


rag_chain = prompt | llm | StrOutputParser()

prompt = PromptTemplate(
    template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|> You are a grader assessing relevance 
    of a retrieved document to a user question. If the document contains keywords related to the user question, 
    grade it as relevant. It does not need to be a stringent test. The goal is to filter out erroneous retrievals. \n
    Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question. \n
    Provide the binary score as a JSON with a single key 'score' and no premable or explaination.
     <|eot_id|><|start_header_id|>user<|end_header_id|>
    Here is the retrieved document: \n\n {document} \n\n
    Here is the user question: {question} \n <|eot_id|><|start_header_id|>assistant<|end_header_id|>
    """,
    input_variables=["question", "document"],
)

retrieval_grader = prompt | llm | JsonOutputParser()

prompt = PromptTemplate(
    template=""" <|begin_of_text|><|start_header_id|>system<|end_header_id|> You are a grader assessing whether 
    an answer is grounded in / supported by a set of facts. Give a binary 'yes' or 'no' score to indicate 
    whether the answer is grounded in / supported by a set of facts. Provide the binary score as a JSON with a 
    single key 'score' and no preamble or explanation. <|eot_id|><|start_header_id|>user<|end_header_id|>
    Here are the facts:
    \n ------- \n
    {documents} 
    \n ------- \n
    Here is the answer: {generation}  <|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
    input_variables=["generation", "documents"],
)

hallucination_grader = prompt | llm | JsonOutputParser()

prompt = PromptTemplate(
    template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|> You are a grader assessing whether an 
    answer is useful to resolve a question. Give a binary score 'yes' or 'no' to indicate whether the answer is 
    useful to resolve a question. Provide the binary score as a JSON with a single key 'score' and no preamble or explanation.
     <|eot_id|><|start_header_id|>user<|end_header_id|> Here is the answer:
    \n ------- \n
    {generation} 
    \n ------- \n
    Here is the question: {question} <|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
    input_variables=["generation", "question"],
)
answer_grader = prompt | llm | JsonOutputParser()

prompt_intent = PromptTemplate(
    template="""
    <|begin_of_text|><|start_header_id|>system<|end_header_id|>You are a classifier assessing whether the query given is just asking a
    question based on the context or if it is giving a command. Give either 'command' or 'question_answering' to indicate whether
    the given query by the user. Provide the intent output in a single key 'intent' and no preamble or explanation.  <|eot_id|><|start_header_id|>user<|end_header_id|> Here is the question/query:
    \n\n
    {question}
    \n\n
    """,
    input_variables=["question"],
)

intent_classifier = prompt_intent | llm | JsonOutputParser()

web_search_tool = TavilySearchResults(k=3)
