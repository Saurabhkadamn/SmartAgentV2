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

st.title("🧠 Custom RAG App with Groq")

# File uploader for PDF, DOCX, XLSX
uploaded_file = st.file_uploader(
    "Upload a document (PDF, DOCX, XLSX)", type=["pdf", "docx", "xlsx"]
)

# User-defined system prompt
system_prompt = st.text_area(
    "System Prompt", value="You are a helpful assistant.", height=100
)

# User query
query = st.text_input("Ask your question:")

if uploaded_file and query and system_prompt:
    with st.spinner("Processing document and generating response..."):
        try:
            # Load document text
            documents = load_file(uploaded_file)
            
            if not documents or not documents[0].strip():
                st.error("No text could be extracted from the uploaded file.")
                st.stop()

            # Create embeddings and build FAISS vector store
            embed_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
            db = FAISS.from_texts(documents, embed_model)
            retriever = db.as_retriever()

            # Instantiate the Groq LLM
            groq_api_key = os.getenv("GROQ_API_KEY")
            if not groq_api_key:
                st.error("GROQ_API_KEY not found in environment variables.")
                st.stop()
                
            llm = ChatGroq(api_key=groq_api_key, model_name="llama-3.3-70b-versatile")

            # Set up conversational retrieval chain with memory
            memory = ConversationBufferMemory(
                memory_key="chat_history", 
                return_messages=True,
                output_key="answer"
            )
            
            # Create a custom prompt that includes the system prompt
            custom_prompt = PromptTemplate(
                input_variables=["context", "question"],
                template=f"""{system_prompt}

Context: {{context}}

Question: {{question}}

Answer:"""
            )
            
            chain = ConversationalRetrievalChain.from_llm(
                llm=llm,
                retriever=retriever,
                memory=memory,
                combine_docs_chain_kwargs={"prompt": custom_prompt},
                return_source_documents=True
            )

            # Run the chain and display result
            result = chain({"question": query})
            
            st.write("### Answer:")
            st.write(result["answer"])
            
            # Optionally show source documents
            if st.checkbox("Show source context"):
                st.write("### Source Context:")
                for i, doc in enumerate(result.get("source_documents", [])):
                    st.write(f"**Source {i+1}:**")
                    st.write(doc.page_content[:500] + "..." if len(doc.page_content) > 500 else doc.page_content)
                    
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            st.write("Please check your API key and try again.")