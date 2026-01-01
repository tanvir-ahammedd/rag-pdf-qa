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
from langchain_core.documents import Document
from dotenv import load_dotenv

# OCR libraries
try:
    import pytesseract
    from pdf2image import convert_from_path
    from PIL import Image
    import fitz  # PyMuPDF
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False

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

def extract_text_with_ocr(pdf_path):
    """Extract text from PDF using OCR (for scanned/image-based PDFs)"""
    if not OCR_AVAILABLE:
        st.warning("OCR libraries not available. Install: pytesseract, pdf2image, PyMuPDF, Pillow")
        return []
    
    documents = []
    try:
        # Set poppler path for different environments
        poppler_path = None
        if os.path.exists('/usr/bin'):
            poppler_path = '/usr/bin'
        
        if poppler_path:
            images = convert_from_path(pdf_path, dpi=300, poppler_path=poppler_path)
        else:
            images = convert_from_path(pdf_path, dpi=300)
        
        for page_num, image in enumerate(images, start=1):
            # Perform OCR on the image
            text = pytesseract.image_to_string(image, lang='eng')
            
            if text.strip():
                doc = Document(
                    page_content=text,
                    metadata={
                        "source": os.path.basename(pdf_path),
                        "page": page_num,
                        "extraction_method": "OCR"
                    }
                )
                documents.append(doc)
        
        return documents
    except Exception as e:
        st.error(f"OCR extraction failed: {str(e)}")
        st.info("💡 Make sure 'packages.txt' file exists with required system packages")
        return []

def is_scanned_pdf(pdf_path):
    """Check if PDF is scanned (image-based) or has extractable text"""
    try:
        doc = fitz.open(pdf_path)
        total_chars = 0
        
        # Check first 3 pages
        for page_num in range(min(3, len(doc))):
            page = doc[page_num]
            text = page.get_text()
            total_chars += len(text.strip())
        
        doc.close()
        
        return total_chars < 50
    except:
        return False

def load_pdf_with_fallback(pdf_path):
    """Load PDF with automatic fallback to OCR if needed"""
    documents = []
    
    try:
        loader = PyPDFLoader(pdf_path)
        docs = loader.load()
        
        total_text = sum(len(doc.page_content.strip()) for doc in docs)
        
        if total_text < 100:
            st.info(f"📄 Detected scanned PDF: {os.path.basename(pdf_path)}. Applying OCR...")
            if OCR_AVAILABLE:
                documents = extract_text_with_ocr(pdf_path)
                if documents:
                    st.success(f"✅ OCR completed for {os.path.basename(pdf_path)}")
                else:
                    st.warning(f"⚠️ OCR failed for {os.path.basename(pdf_path)}")
            else:
                st.error("OCR libraries not installed. Cannot process scanned PDFs.")
        else:
            documents = docs
            
    except Exception as e:
        st.warning(f"Standard extraction failed for {os.path.basename(pdf_path)}. Trying OCR...")
        if OCR_AVAILABLE:
            documents = extract_text_with_ocr(pdf_path)
        else:
            st.error(f"Error loading PDF: {str(e)}")
    
    return documents

def create_vector_embedding(uploaded_files, use_ocr=False):
    """Create vector embeddings from uploaded PDF documents"""
    if uploaded_files:
        with st.spinner("Processing documents and creating embeddings..."):
            try:
                st.session_state.embeddings = HuggingFaceEmbeddings(
                    model_name="sentence-transformers/all-MiniLM-L6-v2"
                )
                st.session_state.docs = []
                
                for uploaded_file in uploaded_files:
                    # Save uploaded file to temporary location
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                        tmp_file.write(uploaded_file.getvalue())
                        tmp_file_path = tmp_file.name
                    
                    # Load PDF with automatic OCR fallback
                    if use_ocr:
                        docs = load_pdf_with_fallback(tmp_file_path)
                    else:
                        loader = PyPDFLoader(tmp_file_path)
                        docs = loader.load()
                    
                    st.session_state.docs.extend(docs)
                    
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


# Page config
st.set_page_config(
    page_title="RAG Q&A with OCR",
    page_icon="📚",
    layout="wide"
)

st.title("📚 RAG Document Q&A System with OCR")
st.markdown("""
This application uses **Retrieval-Augmented Generation (RAG)** to answer questions from your documents.
- Powered by **Groq** and **Llama3**
- Uses **FAISS** for vector storage
- Embeddings with **HuggingFace (MiniLM)**
- ✨ **OCR Support** for scanned/image-based PDFs
""")

# Sidebar
with st.sidebar:
    st.header("⚙️ Setup")
    
    # OCR status
    if OCR_AVAILABLE:
        st.success("✅ OCR Enabled")
    else:
        st.warning("⚠️ OCR Not Available")
        with st.expander("Enable OCR"):
            st.code("""
pip install pytesseract
pip install pdf2image
pip install PyMuPDF
pip install Pillow

# Also install Tesseract OCR:
# Ubuntu: sudo apt-get install tesseract-ocr
# macOS: brew install tesseract
# Windows: Download from GitHub
            """)
    
    st.markdown("""
    ### Steps:
    1. Upload PDF files below
    2. Enable OCR if needed
    3. Click **'Create Vector Database'**
    4. Ask questions about your documents
    """)
    
    uploaded_files = st.file_uploader(
        "Upload PDF files",
        type=['pdf'],
        accept_multiple_files=True,
        help="Upload one or more PDF files (supports scanned PDFs with OCR)"
    )
    
    use_ocr = st.checkbox(
        "Enable OCR Detection",
        value=True,
        help="Automatically detect and process scanned PDFs using OCR",
        disabled=not OCR_AVAILABLE
    )
    
    if st.button("Create Vector Database", use_container_width=True):
        if uploaded_files:
            if create_vector_embedding(uploaded_files, use_ocr):
                st.success("✅ Vector database created successfully!")
            else:
                st.error("❌ Failed to create vector database")
        else:
            st.warning("⚠️ Please upload PDF files first")
    
    st.markdown("---")
    st.markdown("### 📊 Status")
    if "vectors" in st.session_state:
        st.success("✅ Database Ready")
        st.info(f"📄 Documents: {len(st.session_state.docs)}")
        st.info(f"🔢 Chunks: {len(st.session_state.final_documents)}")
        
        # Show extraction methods used
        ocr_docs = sum(1 for doc in st.session_state.docs 
                       if doc.metadata.get('extraction_method') == 'OCR')
        if ocr_docs > 0:
            st.info(f"🔍 OCR pages: {ocr_docs}")
    else:
        st.warning("⚠️ Database not initialized")

# Main content
col1, col2 = st.columns([3, 1])

with col1:
    user_prompt = st.text_input(
        "Enter your question:",
        placeholder="e.g., What are the main findings?"
    )

with col2:
    st.write("")

if user_prompt:
    if "vectors" not in st.session_state:
        st.warning("⚠️ Please create the vector database first using the button in the sidebar.")
    else:
        with st.spinner("🤔 Thinking..."):
            try:
                document_chain = create_stuff_documents_chain(llm, prompt)
                retriever = st.session_state.vectors.as_retriever()
                retrieval_chain = create_retrieval_chain(retriever, document_chain)
                
                start = time.process_time()
                response = retrieval_chain.invoke({'input': user_prompt})
                response_time = time.process_time() - start
                
                # Display answer
                st.markdown("### 💡 Answer:")
                st.markdown(response['answer'])
                
                st.caption(f"⏱️ Response time: {response_time:.2f} seconds")
                
                # Display source documents
                with st.expander("📄 View Source Documents"):
                    for i, doc in enumerate(response['context']):
                        st.markdown(f"**Source {i+1}:**")
                        
                        # Show metadata
                        if hasattr(doc, 'metadata'):
                            meta = doc.metadata
                            col_a, col_b = st.columns(2)
                            with col_a:
                                st.caption(f"📁 File: {meta.get('source', 'Unknown')}")
                            with col_b:
                                st.caption(f"📄 Page: {meta.get('page', 'N/A')}")
                            
                            if meta.get('extraction_method') == 'OCR':
                                st.caption("🔍 Extracted via OCR")
                        
                        st.text(doc.page_content)
                        st.markdown("---")
                        
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p>Built with Streamlit, LangChain, Groq, Huggingface, and Tesseract OCR</p>
</div>
""", unsafe_allow_html=True)