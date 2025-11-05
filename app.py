import os
import streamlit as st
from langchain_community.document_loaders import JSONLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

# ========================
# CARGA DE VARIABLES .ENV
# ========================
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# ========================
# PROCESAMIENTO RAG
# ========================
@st.cache_resource
def load_and_process_documents():
    try:
        loader = JSONLoader(
            file_path="./data.jsonl",
            jq_schema=".texto_completo",
            text_content=False,
            json_lines=True
        )
        docs = loader.load()

        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=120)
        chunks = splitter.split_documents(docs)

        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vstore = FAISS.from_documents(chunks, embeddings)
        return vstore

    except Exception as e:
        st.error(f"Error cargando base documental: {e}")
        return None

def clear_query():
    st.session_state["user_query_input"] = ""

# ========================
# INTERFAZ PRINCIPAL
# ========================
def main():
    st.set_page_config(page_title="Asistente Normativo MINAM", layout="wide")

    # ===== CSS personalizado =====
    st.markdown("""
        <style>
        body {
            background-color: #0B1215;
        }
        .stApp {
            background: #0B1215;
            color: #F4F6F6;
        }

        /* Header Institucional */
        .header {
            background-color: #02733E;
            padding: 18px;
            text-align: center;
            color: white;
            border-bottom: 6px solid #D22630;
            font-size: 24px;
            font-weight: bold;
        }

        /* Sidebar */
        [data-testid="stSidebar"] {
            background-color: #0E2421;
            padding: 0 !important;
        }
        .sidebar-box {
            padding: 12px;
            color: #F4F6F6;
        }
        .sidebar-box h3 {
            color: #E6F8E0;
        }

        .image-banner img {
            width: 100%;
            height: 180px;
            object-fit: cover;
            border-bottom: 4px solid #D22630;
        }

        /* Caja de respuesta */
        .response-box {
            background: #113A2D;
            border-left: 5px solid #D22630;
            padding: 15px;
            border-radius: 8px;
            font-size: 16px;
        }

        hr {
            border: 1px solid #D22630;
        }
        </style>
    """, unsafe_allow_html=True)

    # ===== HEADER =====
    st.markdown('<div class="header">Asistente RAG Normativo | Ministerio del Ambiente – Perú</div>', unsafe_allow_html=True)

    # ===== SIDEBAR =====
    with st.sidebar:
        st.markdown("""
        <div class="image-banner">
            <img src="https://www.gob.pe/assets/logo_peru.png">
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="sidebar-box">
            <h3>MINAM – Perú</h3>
            <p>Consultas sobre normativa ambiental, resoluciones oficiales y gestión pública.</p>
            <hr>
            <p>Motor: Gemini 2.5 Flash + RAG</p>
        </div>
        """, unsafe_allow_html=True)

    # ===== CONTENIDO =====
    st.title("🌿 Asesor Normativo Ambiental")
    st.write("Base documental: Resoluciones y normas del MINAM.")

    vstore = load_and_process_documents()
    if not vstore:
        st.stop()

    if not GEMINI_API_KEY:
        st.error("Configurar GEMINI_API_KEY en .env")
        st.stop()

    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.0, google_api_key=GEMINI_API_KEY)
    retriever = vstore.as_retriever(search_kwargs={"k": 5})

    prompt = ChatPromptTemplate.from_template("""
        Responde como asesor jurídico ambiental del MINAM Perú.
        Usa exclusivamente el contexto entregado y sé formal.

        CONTEXTO:
        {context}

        PREGUNTA:
        {question}
    """)

    def format_docs(d):
        return "\n\n".join(x.page_content for x in d)

    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt | llm | StrOutputParser()
    )

    with st.form("consulta"):
        consulta = st.text_input("Formula tu consulta normativa", key="user_query_input")
        submit = st.form_submit_button("Consultar")
        st.form_submit_button("Limpiar", on_click=clear_query)

        if submit and consulta:
            with st.spinner("Buscando en normativa..."):
                respuesta = chain.invoke(consulta)
                fuentes = retriever.invoke(consulta)

            st.subheader("✅ Respuesta")
            st.markdown(f"<div class='response-box'>{respuesta}</div>", unsafe_allow_html=True)

            st.subheader("📚 Documentos utilizados")
            for d in fuentes:
                st.expander(d.metadata.get("titulo", "Documento")).write(d.page_content)

if __name__ == "__main__":
    main()
