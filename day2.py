import os
import bs4
from langchain_chroma import Chroma
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
import requests

RETRIEVE_COUNT = 6

with open("open-ai-key", "r") as file:
    os.environ["OPENAI_API_KEY"] = file.read().strip()

session = requests.Session()
session.headers.update({
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
})

url_list = ['2023-06-23-agent/', '2023-03-15-prompt-engineering/', '2023-10-25-adv-attack-llm/']
web_docs = {}
# 모든 문서의 splits를 저장할 리스트 생성
all_splits = []

for paths in url_list:
    loader = WebBaseLoader(
        web_paths=("https://lilianweng.github.io/posts/" + paths,),
        session=session,
        bs_kwargs=dict(
            parse_only=bs4.SoupStrainer(
                class_=("post-content", "post-title", "post-header")
            )
        ),
    )

    web_docs[paths] = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n\n\n"],
        chunk_size=3000,
        chunk_overlap=300,
        length_function=len,
        is_separator_regex=False
    )
    splits = text_splitter.split_documents(web_docs[paths])
    all_splits.extend(splits)

db = Chroma.from_documents(
    documents=all_splits,
    embedding=OpenAIEmbeddings(model="text-embedding-3-small")
)

retriever = db.as_retriever(
    search_type="similarity",
    search_kwargs={"k": RETRIEVE_COUNT}
)

llm = ChatOpenAI(model="gpt-4o-mini")
parser = JsonOutputParser()


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

prompt = PromptTemplate(
    template="""
        Your job is to check if the `query` is relevant to the `docs`.  
        Output your answer in JSON format with a single key 'relevance' and value either 'yes' or 'no'
        {format_instructions}
        query: {query}
        docs: {docs}
    """,
    input_variables=["query", "docs"],
    partial_variables={"format_instructions": parser.get_format_instructions()},
)

rag_chain = (
    {"docs": retriever | format_docs, "query": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)


def retrieve_relevance(query):
    print("============================================")
    docs = retriever.invoke(query)
    print(f"Retrieved {len(docs)} documents")
    for i, doc in enumerate(docs):
        print("-----------------------------------------")
        print(f"Document {i + 1}: {doc.metadata['source']}")
        # print(f"Content: {doc.page_content[:150]}...\n")
        print(f"Content: {doc.page_content}\n")

        chain = prompt | llm | parser
        print("answer:", chain.invoke({"query": query, "docs": doc.page_content}))

def retrieve_relevance_chain(query):
    for chunk in rag_chain.stream(query):
        print(chunk, end="", flush=True)

if __name__ == "__main__":
    retrieve_relevance("agent memory")
    retrieve_relevance_chain("What is Task Decomposition?")



