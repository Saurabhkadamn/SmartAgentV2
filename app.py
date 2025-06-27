import streamlit as st
from utils.loaders import load_file

# Embeddings and vector store
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Groq chat model and chain
from langchain_groq import ChatGroq
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure page
st.set_page_config(
    page_title="Custom RAG App",
    page_icon="🧠",
    layout="wide"
)

st.title("🧠 Custom RAG App with Groq")
st.markdown("Upload a document and ask questions about its content using AI-powered retrieval.")

# Initialize session state for conversation history
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'vectorstore' not in st.session_state:
    st.session_state.vectorstore = None
if 'current_file' not in st.session_state:
    st.session_state.current_file = None

# Sidebar for configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Model selection
    model_options = [
        "llama-3.3-70b-versatile",
        "llama-3.1-70b-versatile", 
        "mixtral-8x7b-32768",
        "gemma2-9b-it"
    ]
    selected_model = st.selectbox("Select Model:", model_options)
    
    # Temperature setting
    temperature = st.slider("Temperature:", 0.0, 1.0, 0.7, 0.1)
    
    # Max tokens
    max_tokens = st.number_input("Max Tokens:", 100, 4000, 1000, 50)
    
    # Retrieval settings
    st.subheader("🔍 Retrieval Settings")
    top_k = st.slider("Number of documents to retrieve:", 1, 10, 4)
    
    # Clear conversation button
    if st.button("🗑️ Clear Conversation"):
        st.session_state.chat_history = []
        st.rerun()

# Main content area
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📄 Document Upload")
    uploaded_file = st.file_uploader(
        "Upload a document", 
        type=["pdf", "docx", "xlsx"],
        help="Supported formats: PDF, DOCX, XLSX"
    )
    
    # Process document if new file uploaded
    if uploaded_file and (st.session_state.current_file != uploaded_file.name):
        with st.spinner("Processing document..."):
            try:
                # Load document text
                documents = load_file(uploaded_file)
                
                if not documents or not documents[0].strip():
                    st.error("❌ No text could be extracted from the uploaded file.")
                else:
                    # Create embeddings and build FAISS vector store
                    embed_model = HuggingFaceEmbeddings(
                        model_name="all-MiniLM-L6-v2",
                        model_kwargs={'device': 'cpu'}
                    )
                    
                    # Split text into chunks if it's very long
                    from langchain.text_splitter import RecursiveCharacterTextSplitter
                    text_splitter = RecursiveCharacterTextSplitter(
                        chunk_size=1000,
                        chunk_overlap=200,
                        length_function=len
                    )
                    
                    # If documents is a list of strings, join and split
                    if isinstance(documents, list):
                        full_text = "\n\n".join(documents)
                    else:
                        full_text = documents
                    
                    chunks = text_splitter.split_text(full_text)
                    
                    # Create vector store
                    vectorstore = FAISS.from_texts(chunks, embed_model)
                    st.session_state.vectorstore = vectorstore
                    st.session_state.current_file = uploaded_file.name
                    
                    st.success(f"✅ Document processed successfully! Created {len(chunks)} text chunks.")
                    
            except Exception as e:
                st.error(f"❌ Error processing document: {str(e)}")

with col2:
    st.subheader("💬 System Prompt")
    system_prompt = st.text_area(
        "Customize the AI's behavior:",
        value="You are a helpful assistant that answers questions based on the provided context. Be concise and accurate.",
        height=120,
        help="This prompt will guide how the AI responds to your questions."
    )

# Chat interface
st.subheader("💭 Chat with your Document")

# Display chat history
if st.session_state.chat_history:
    for i, (question, answer) in enumerate(st.session_state.chat_history):
        with st.container():
            st.markdown(f"**You:** {question}")
            st.markdown(f"**AI:** {answer}")
            st.divider()

# Query input
query = st.text_input(
    "Ask your question:",
    placeholder="What is this document about?",
    key="query_input"
)

# Process query
if st.button("Send", type="primary") or (query and st.session_state.get('query_submitted', False)):
    if not uploaded_file:
        st.warning("⚠️ Please upload a document first.")
    elif not query.strip():
        st.warning("⚠️ Please enter a question.")
    elif not st.session_state.vectorstore:
        st.warning("⚠️ Please wait for document processing to complete.")
    else:
        with st.spinner("Generating response..."):
            try:
                # Check for Groq API key
                groq_api_key = os.getenv("GROQ_API_KEY")
                if not groq_api_key:
                    st.error("❌ GROQ_API_KEY not found in environment variables.")
                    st.info("Please add your Groq API key to your .env file.")
                    st.stop()
                
                # Set up retriever
                retriever = st.session_state.vectorstore.as_retriever(
                    search_kwargs={"k": top_k}
                )
                
                # Instantiate the Groq LLM with custom parameters
                llm = ChatGroq(
                    api_key=groq_api_key,
                    model_name=selected_model,
                    temperature=temperature,
                    max_tokens=max_tokens
                )
                
                # Set up memory
                memory = ConversationBufferMemory(
                    memory_key="chat_history",
                    return_messages=True,
                    output_key="answer"
                )
                
                # Add previous conversation to memory
                for q, a in st.session_state.chat_history[-5:]:  # Keep last 5 exchanges
                    memory.chat_memory.add_user_message(q)
                    memory.chat_memory.add_ai_message(a)
                
                # Create a custom prompt
                custom_prompt = PromptTemplate(
                    input_variables=["context", "question"],
                    template=f"""{system_prompt}

Context from the document:
{{context}}

Question: {{question}}

Please provide a helpful and accurate answer based on the context provided. If the context doesn't contain enough information to answer the question, please say so.

Answer:"""
                )
                
                # Create chain
                chain = ConversationalRetrievalChain.from_llm(
                    llm=llm,
                    retriever=retriever,
                    memory=memory,
                    combine_docs_chain_kwargs={"prompt": custom_prompt},
                    return_source_documents=True,
                    verbose=False
                )
                
                # Run the chain
                result = chain({"question": query})
                answer = result["answer"]
                
                # Add to chat history
                st.session_state.chat_history.append((query, answer))
                
                # Clear the input and rerun to show new message
                st.session_state.query_submitted = False
                st.rerun()
                
            except Exception as e:
                st.error(f"❌ An error occurred: {str(e)}")
                if "rate limit" in str(e).lower():
                    st.info("💡 You may have hit the API rate limit. Please wait a moment and try again.")
                elif "api key" in str(e).lower():
                    st.info("💡 Please check your GROQ_API_KEY in the .env file.")

# Footer with usage tips
with st.expander("💡 Usage Tips"):
    st.markdown("""
    **How to use this app:**
    1. **Upload a document** (PDF, DOCX, or XLSX) using the file uploader
    2. **Customize the system prompt** to guide the AI's behavior
    3. **Ask questions** about your document content
    4. **Adjust settings** in the sidebar for different models and parameters
    
    **Tips for better results:**
    - Be specific in your questions
    - Try different temperature settings (lower = more focused, higher = more creative)
    - Adjust the number of retrieved documents based on your needs
    - Use the system prompt to specify the desired response format
    """)

# Display current configuration
if st.checkbox("Show Current Configuration"):
    st.json({
        "model": selected_model,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "retrieval_top_k": top_k,
        "document_loaded": st.session_state.current_file is not None,
        "chat_history_length": len(st.session_state.chat_history)
    })