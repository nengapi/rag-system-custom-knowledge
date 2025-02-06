from typing import List
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama.llms import OllamaLLM
from langchain_community.vectorstores import FAISS
import logging
import psutil
import os


class RAGPipeline:
    def __init__(self, model_name: str = "deepseek-r1:7b", max_memory_gb: float = 3.0):
        self.setup_logging()
        self.check_system_memory(max_memory_gb)

        # Load the language model (LLM)
        self.llm = OllamaLLM(model=model_name, base_url="http://ollama:11434")

        # Initialize embeddings model for text vectorization
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-mpnet-base-v2",
            model_kwargs={"device": "cpu"},  # Use CPU for efficiency
        )

        # Define the RAG prompt template for English language responses
        self.prompt = ChatPromptTemplate.from_template(
            """
        Answer questions using only the information provided in the given context.
        If the answer cannot be found in the provided context, respond with "Unable to answer this question from the available information."
        
        Context: {context}
        Question: {question}
        Answer: """
        )

    def setup_logging(self):
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def check_system_memory(self, max_memory_gb: float):
        available_memory = psutil.virtual_memory().available / (1024**3)
        self.logger.info(f"Available system memory: {available_memory:.1f} GB")
        if available_memory < max_memory_gb:
            self.logger.warning("Memory is below recommended threshold.")

    def load_and_split_documents(self, file_path: str) -> List[Document]:
        loader = TextLoader(file_path)
        documents = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            length_function=len,
            add_start_index=True,
        )
        splits = text_splitter.split_documents(documents)
        self.logger.info(f"Created {len(splits)} document chunks")
        return splits

    def create_vectorstore(self, documents: List[Document]) -> FAISS:
        batch_size = 32
        vectorstore = FAISS.from_documents(documents[:batch_size], self.embeddings)

        for i in range(batch_size, len(documents), batch_size):
            batch = documents[i : i + batch_size]
            vectorstore.add_documents(batch)
            self.logger.info(f"Processed batch {i//batch_size + 1}")
        return vectorstore

    def setup_rag_chain(self, vectorstore: FAISS):
        retriever = vectorstore.as_retriever(
            search_type="similarity", search_kwargs={"k": 2, "fetch_k": 3}
        )

        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)

        rag_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | self.prompt
            | self.llm
            | StrOutputParser()
        )

        return rag_chain

    def query(self, chain, question: str) -> str:
        memory_usage = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
        self.logger.info(f"Memory usage: {memory_usage:.1f} MB")
        return chain.invoke(question)


def main(question: str):
    rag = RAGPipeline()

    documents = rag.load_and_split_documents("data/knowledge.txt")
    vectorstore = rag.create_vectorstore(documents)
    chain = rag.setup_rag_chain(vectorstore)

    response = rag.query(chain, question)
    return response


if __name__ == "__main__":
    main()
