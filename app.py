import nest_asyncio
nest_asyncio.apply()

import os
import streamlit as st
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import WebBaseLoader, WikipediaLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq  # <-- NEW

# --- Page Configuration ---
st.set_page_config(
    page_title="AI Fitness Coach",
    page_icon="💪",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS ---
def load_css():
    st.markdown("""<style>
        .stApp {
            background-image: url('https://images.unsplash.com/photo-1581009137042-c552e485697a?q=80&w=2070&auto=format&fit=crop');
            background-size: cover;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }
        .stApp > div:first-child > div > div > div > div { color: #ffffff; }
        .stChatMessage {
            border-radius: 10px;
            padding: 1rem;
            margin-bottom: 1rem;
            box-shadow: 0 4px 8px rgba(0,0,0,0.3);
            border: 1px solid #44475a;
            background-color: rgba(40, 42, 54, 0.85);
            color: #f8f8f2;
        }
        [data-testid="stChatMessageContent"] p { color: #f8f8f2; }
        [data-testid="stSidebar"] {
            background-color: rgba(26, 26, 26, 0.9);
            border-right: 1px solid #44475a;
        }
        [data-testid="stSidebar"] .st-emotion-cache-16txtl3 { color: #f8f8f2; }
        [data-testid="stSidebar"] .stButton>button {
            width: 100%;
            border-radius: 8px;
            background-color: #ff4b4b;
            color: white;
            border: none;
            padding: 10px 0;
            transition: background-color 0.3s ease;
        }
        [data-testid="stSidebar"] .stButton>button:hover {
            background-color: #e03a3a;
        }
        h1, h2 { color: #ffffff; text-shadow: 2px 2px 4px rgba(0,0,0,0.5); }
    </style>""", unsafe_allow_html=True)

# --- Load Environment Variables ---
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# --- Document Loader ---
def get_documents_from_urls(urls):
    all_docs = []
    for url in urls:
        try:
            if "wikipedia.org" in url:
                query = url.split("/")[-1]
                loader = WikipediaLoader(query=query, load_max_docs=1, doc_content_chars_max=20000)
            else:
                loader = WebBaseLoader(url)
            docs = loader.load()
            all_docs.extend(docs)
        except Exception as e:
            st.warning(f"Could not load content from {url}: {e}")
    return all_docs

# --- Create Vector Store ---
def create_vector_store_from_websites(urls):
    docs = get_documents_from_urls(urls)
    if not docs:
        return None
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    split_docs = text_splitter.split_documents(docs)

    # Correct embedding model
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = FAISS.from_documents(split_docs, embedding=embeddings)
    return vector_store

# --- Create RAG Chain ---
def create_rag_chain(vector_store):
    llm = ChatGroq(
        groq_api_key=GROQ_API_KEY,
        model="llama3-8b-8192",  # or llama3-70b-8192
        temperature=0.7
    )

    prompt_template = """
    You are an encouraging and knowledgeable fitness assistant.
    Your goal is to provide safe, helpful, and motivational fitness advice based ONLY on the following context.
    If the answer is not in the context, clearly state that you don't have enough information from the provided source.
    Please provide the answer in the same language as the user's question.

    Context:
    {context}

    Question:
    {input}
    """
    prompt = ChatPromptTemplate.from_template(prompt_template)
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)
    return rag_chain

# --- Main Application ---
def main():
    load_css()

    with st.sidebar:
        st.header("Control Panel")
        st.write("Teach the AI by providing URLs from trusted fitness websites.")
        default_urls = "https://en.wikipedia.org/wiki/Strength_training\nhttps://en.wikipedia.org/wiki/Physical_fitness"
        website_urls_text = st.text_area("Enter Website URLs (one per line)", default_urls, height=150)

        if st.button("Index Websites"):
            urls = [url.strip() for url in website_urls_text.split('\n') if url.strip()]
            if not urls:
                st.warning("Please enter at least one URL.")
            else:
                with st.spinner("Scraping and indexing..."):
                    try:
                        vector_store = create_vector_store_from_websites(urls)
                        if vector_store:
                            st.session_state.rag_chain = create_rag_chain(vector_store)
                            st.success("Knowledge base is ready!")
                        else:
                            st.error("Failed to load content. Please check URLs.")
                    except Exception as e:
                        st.error(f"An error occurred: {e}")

    st.title("💪 AI Fitness Coach")
    st.write("Your personal AI-powered coach. Use the control panel on the left to teach me, then ask your questions below!")

    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "Hello! How can I help you on your fitness journey today?"}]

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Ask a fitness question..."):
        if "rag_chain" not in st.session_state or st.session_state.rag_chain is None:
            st.warning("Please index a website using the control panel before asking questions.")
        else:
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    response = st.session_state.rag_chain.invoke({"input": prompt})
                    st.markdown(response["answer"])

            st.session_state.messages.append({"role": "assistant", "content": response["answer"]})

if __name__ == '__main__':
    main()
