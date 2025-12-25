import streamlit as st
import os
import time
import tempfile
from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from dotenv import load_dotenv


load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY") or st.secrets.get("GROQ_API_KEY")

if not groq_api_key:
    st.error("GROQ_API_KEY not found. Please set it in secrets or .env file.")
    st.stop()

llm = ChatGroq(groq_api_key=groq_api_key, model_name="llama-3.1-8b-instant")

prompt = ChatPromptTemplate.from_template(
    """
    Answer the questions based on the provided context only.
    Please provide the most accurate response based on the question.
    
    <context>
    {context}
    </context>
    
    Question: {input}
    
    Answer:
    """
)

def create_vector_embedding(uploaded_files):
    """Create vector embeddings from uploaded PDF documents"""
    if uploaded_files:
        with st.spinner("Processing documents and creating embeddings..."):
            try:
                st.session_state.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
                st.session_state.docs = []
                
                for uploaded_file in uploaded_files:
                    # Save uploaded file to temporary location
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                        tmp_file.write(uploaded_file.getvalue())
                        tmp_file_path = tmp_file.name
                    
                    loader = PyPDFLoader(tmp_file_path)
                    docs = loader.load()
                    st.session_state.docs.extend(docs)
                    
                    # Clean up temporary file
                    os.unlink(tmp_file_path)
                
                if not st.session_state.docs:
                    st.error("No documents loaded. Please check your PDF files.")
                    return False
                
                st.session_state.text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000,
                    chunk_overlap=200
                )
                st.session_state.final_documents = st.session_state.text_splitter.split_documents(
                    st.session_state.docs[:50]
                )
                st.session_state.vectors = FAISS.from_documents(
                    st.session_state.final_documents,
                    st.session_state.embeddings
                )
                return True
            except Exception as e:
                st.error(f"Error creating embeddings: {str(e)}")
                return False
    return False


st.title("RAG Document Q&A System")
st.markdown("""
This application uses **Retrieval-Augmented Generation (RAG)** to answer questions from your research papers.
- Powered by **Groq** and **Llama3**
- Uses **FAISS** for vector storage
- Embeddings with **HuggingFace (MiniLM)**
""")

with st.sidebar:
    st.header("Setup")
    st.markdown("""
    ### Steps:
    1. Upload PDF files below
    2. Click **'Create Vector Database'** button
    3. Ask questions about your documents
    """)
    
    uploaded_files = st.file_uploader(
        "Upload PDF files",
        type=['pdf'],
        accept_multiple_files=True,
        help="Upload one or more PDF files"
    )
    
    if st.button("Create Vector Database", use_container_width=True):
        if uploaded_files:
            if create_vector_embedding(uploaded_files):
                st.success("Vector database created successfully!")
            else:
                st.error("Failed to create vector database")
        else:
            st.warning("Please upload PDF files first")
    
    st.markdown("---")
    st.markdown("### Status")
    if "vectors" in st.session_state:
        st.success("Database Ready")
        st.info(f"Documents loaded: {len(st.session_state.docs)}")
        st.info(f"Chunks created: {len(st.session_state.final_documents)}")
    else:
        st.warning("Database not initialized")

# Main content
user_prompt = st.text_input("Enter your question:", placeholder="e.g., What are the main findings?")

if user_prompt:
    if "vectors" not in st.session_state:
        st.warning("Please create the vector database first using the button in the sidebar.")
    else:
        with st.spinner("Thinking..."):
            try:
                document_chain = create_stuff_documents_chain(llm, prompt)
                retriever = st.session_state.vectors.as_retriever()
                retrieval_chain = create_retrieval_chain(retriever, document_chain)
                
                start = time.process_time()
                response = retrieval_chain.invoke({'input': user_prompt})
                response_time = time.process_time() - start
                
                # Display answer
                st.markdown("### Answer:")
                st.markdown(response['answer'])
                
                st.caption(f"Response time: {response_time:.2f} seconds")
                
                # Display source documents
                with st.expander("View Source Documents"):
                    for i, doc in enumerate(response['context']):
                        st.markdown(f"**Source {i+1}:**")
                        st.text(doc.page_content)
                        st.markdown("---")
                        
            except Exception as e:
                st.error(f"Error: {str(e)}")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p>Built with Streamlit, LangChain, Groq, and Huggingface</p>
</div>
""", unsafe_allow_html=True)