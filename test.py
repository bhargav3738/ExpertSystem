
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain.document_loaders import TextLoader

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from langchain_core.prompts.chat import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from dotenv import load_dotenv
import os


def ragResponse(file):
    load_dotenv(override=True)
    documents = TextLoader(file).load()

   
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(documents)
    vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings(openai_api_key=API_KEY))
    retriever = vectorstore.as_retriever()
    
    return retriever

API_KEY=os.getenv("OPEN_API_KEY")

system_prompt = (
"You are an assistant for question-answering tasks. "
"Use the following pieces of retrieved context to answer "
"the question. If you don't know the answer, say that you "
"don't know. Use three sentences maximum and keep the "
"answer concise."
"\n\n"
"{context}"
)

prompt = ChatPromptTemplate.from_messages(
[
("system", system_prompt),
("human", "{input}"),
]
)

llm = ChatOpenAI(
model="gpt-3.5-turbo",
temperature=0.5,
api_key=API_KEY
)

file ="BusinessContext.txt"
# print(ragResponse(file))

retriever = ragResponse(file)

qa_chain=create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, qa_chain)

def invoker(prompt):
    response = rag_chain.invoke({"input": prompt})
    return response["answer"]
    
import streamlit as st

# Define the Streamlit app
def main():
    st.title("RAG Chain Interface")
    
    # User input
    user_input = st.text_input("Enter your question:")
    
    if st.button("Submit"):
        # Invoke the rag_chain with user input
        response = invoker(user_input)
        
        # Print the output
        st.write("Answer:", response)

# Run the Streamlit app
if __name__ == "__main__":
    main()