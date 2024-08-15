
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.document_loaders import TextLoader
from langchain_community.document_loaders import DirectoryLoader
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import OpenAIEmbeddings
from langchain_core.prompts.chat import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from dotenv import load_dotenv
from langchain.chains import create_retrieval_chain
import os
import atexit
import streamlit as st


load_dotenv(override=True)
API_KEY=os.getenv("OPEN_API_KEY")

# Making Data Folder
DATA_PATH = "data/"
os.makedirs(DATA_PATH, exist_ok=True)

# Function to list files in the data folder
def list_files():
    return os.listdir(DATA_PATH)

# Function to remove all files in the data folder
def clear_data_folder():
    for file_name in os.listdir(DATA_PATH):
        file_path = os.path.join(DATA_PATH, file_name)
        if os.path.isfile(file_path):
            os.remove(file_path)
            

# Register the cleanup function to be called when the app stops running
atexit.register(clear_data_folder)






uploaded_files = st.sidebar.file_uploader("Choose files", accept_multiple_files=True)

if uploaded_files:
    for uploaded_file in uploaded_files:
        # Ensure the uploaded_file is not None and is a file-like object
        if uploaded_file is not None:
            # Now we safely assume uploaded_file has the attribute 'name'
            file_path = os.path.join(DATA_PATH, uploaded_file.name)
            try:
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
            except IOError as e:
                st.sidebar.error(f"Error saving {uploaded_file.name}: {str(e)}")
            else:
                st.sidebar.success(f"Saved {uploaded_file.name}")
else:
    st.sidebar.warning("No files uploaded.")

    
    
# Display the list of files in the data folder
st.sidebar.subheader("Files in data folder:")
files = list_files()
if files:
    for file in files:
        st.sidebar.write(file)
else:
    st.sidebar.write("No files found.")
    







## STREAMLIT INTERFACE

if not os.listdir(DATA_PATH):
    st.warning("Upload a text file to begin")

else: 
    os.environ["OPENAI_API_KEY"] = API_KEY
    

    # LOAD
    loader = DirectoryLoader(DATA_PATH)
    documents = TextLoader(file).load()

    # SPLIT
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(documents)
    

    # VECTORSTORE
    vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings(openai_api_key=API_KEY))
    retriever = vectorstore.as_retriever()

    # RAG
    # retriever = vectorstore.as_retriever()
    
    st.title("READ MY File📄")
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
    qa_chain=create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, qa_chain)
    
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Ask a question based on your documents?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            response = rag_chain.invoke({"input": prompt})['answer']
            st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})