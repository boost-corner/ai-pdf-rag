from dotenv import load_dotenv

load_dotenv()

from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings

from langchain_core.prompts import PromptTemplate

from langchain_core.vectorstores import InMemoryVectorStore
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_core.documents import Document
from typing_extensions import List, TypedDict
from langgraph.graph import END, StateGraph

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)

embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
vector_store = InMemoryVectorStore(embeddings)


class RagState(TypedDict):
    question: str
    query: str
    docs: List[Document]
    answer: str

# Prompts

rag_prompt_template = PromptTemplate(
    template="""
You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question.
If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.
Question: {question} 
Context: {context} 
Answer:
""",
    input_variables=["question", "context"],
)

query_prompt_template = PromptTemplate(
    template="""
You are an assistant that extracts a concise and effective search query from a given natural language question.
Your task is to understand the core intent of the question and transform it into a short, precise search query that can be used in a search engine.
Only output the search query, nothing else.
Question: {question}
Search query to run:
""",
    input_variables=["question"],
)

# Step implementations
def analyze(state: RagState):
    query_prompt = query_prompt_template.invoke({"question": state["question"]})
    response = llm.invoke(query_prompt)
    return {"query": response.content}


def retrieve(state: RagState):
    retrieved_docs = vector_store.similarity_search(state["query"])
    return {"docs": retrieved_docs}


def generate(state: RagState):
    context = "\n\n".join(doc.page_content for doc in state["docs"])
    question_prompt = rag_prompt_template.invoke({"question": state["question"], "context": context})
    response = llm.invoke(question_prompt)
    return {"answer": response.content}


workflow = StateGraph(RagState)
workflow.add_node("analyze", analyze)
workflow.add_node("retrieve", retrieve)
workflow.add_node("generate", generate)

workflow.set_entry_point("analyze")
workflow.add_edge("analyze", "retrieve")
workflow.add_edge("retrieve", "generate")
workflow.add_edge("generate", END)

# Compile the app
app = workflow.compile()

# Example usage
if __name__ == "__main__":
    file_path = "https://arxiv.org/pdf/2312.10997"

    loader = PyPDFLoader(file_path=file_path)

    docs = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    text_splits = text_splitter.split_documents(docs)

    vector_store.add_documents(documents=text_splits)

    question = input("Enter a question: ")
    result = app.invoke({"question": question})

    print(result["answer"])
