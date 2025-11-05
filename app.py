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

    # ===== CSS Claro & Navbar =====
    st.markdown("""
        <style>
        body {
            background-color: #F6F9F9;
        }
        .stApp {
            background: #F6F9F9;
            color: #1D3B32;
        }

        /* Navbar */
        .navbar {
            background-color: #1C7C54;
            width: 100%;
            padding: 14px;
            color: white;
            font-size: 21px;
            text-align: center;
            border-bottom: 5px solid #145E3A;
            font-weight: bold;
        }

        /* Banner */
        .banner img {
            width: 100%;
            height: 180px;
            object-fit: cover;
            border-bottom: 3px solid #145E3A;
        }

        /* Caja de respuesta */
        .response-box {
            background: #FFFFFF;
            border-left: 4px solid #1C7C54;
            padding: 14px;
            border-radius: 8px;
            font-size: 16px;
            line-height: 1.5;
        }

        /* Sugerencias */
        .suggestion-box {
            background: #E8F3EE;
            border-left: 4px solid #60A68C;
            padding: 10px;
            border-radius: 6px;
            margin-top: 10px;
        }
        .suggestion-box ul {
            font-size: 14px;
            color: #2B4C42;
        }
        </style>

        <div class='navbar'>Asistente Normativo MINAM Perú</div>
        <div class='banner'>
            <img src="https://portal.mineco.gob.pe/wp-content/uploads/2023/10/minam.webp">
        </div>
    """, unsafe_allow_html=True)

    # Ocultar sidebar
    with st.sidebar: pass

    st.title("🌱 Consulta Normativa Ambiental del MINAM")
    st.write("Sistema de asistencia legal con búsqueda en normativa oficial.")

    vstore = load_and_process_documents()
    if not vstore:
        st.stop()

    if not GEMINI_API_KEY:
        st.error("Falta GEMINI_API_KEY en archivo .env")
        st.stop()

    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.0, google_api_key=GEMINI_API_KEY)
    retriever = vstore.as_retriever(search_kwargs={"k": 5})

    prompt = ChatPromptTemplate.from_template("""
        Responde como asesor jurídico ambiental del MINAM Perú.
        Responde únicamente con fundamentos normativos contenidas en el contexto.

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

    # Formulario
    with st.form("consulta"):
        consulta = st.text_input("📌 Escribe tu consulta", key="user_query_input")

        # Sugerencias
        st.markdown("""
        <div class='suggestion-box'>
        <b>Ejemplos:</b>
        <ul>
        <li>¿Qué norma regula la gestión de residuos sólidos?</li>
        <li>¿Quién supervisa la fiscalización ambiental en el Perú?</li>
        <li>¿Cuáles son los lineamientos del SEIA?</li>
        <li>¿Qué sanciones existen por contaminar un recurso hídrico?</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)

        submit = st.form_submit_button("Consultar")
        st.form_submit_button("Limpiar", on_click=clear_query)

        if submit and consulta:
            with st.spinner("Procesando consulta normativa..."):
                respuesta = chain.invoke(consulta)
                fuentes = retriever.invoke(consulta)

            st.subheader("✅ Respuesta")
            st.markdown(f"<div class='response-box'>{respuesta}</div>", unsafe_allow_html=True)

            st.subheader("📄 Documentos consultados")
            for d in fuentes:
                st.expander(d.metadata.get("titulo", "Documento")).write(d.page_content)

if __name__ == "__main__":
    main()
