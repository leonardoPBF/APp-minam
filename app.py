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

    # ===== Estilos Globales =====
    st.markdown("""
        <style>
        body { background-color: #F6F9F9; }
        .stApp { background: #F6F9F9; color: #1D3B32; }

        /* Navbar */
        .navbar {
            background-color: #1C7C54;
            width: 100%;
            padding: 14px;
            color: white;
            font-size: 20px;
            text-align: center;
            border-bottom: 4px solid #145E3A;
            font-weight: 600;
        }

        /* Banner */
        .banner img {
            width: 100%;
            height: 180px;
            object-fit: cover;
        }

        /* Caja respuesta */
        .response-box {
            background: #FFFFFF;
            border-left: 4px solid #1C7C54;
            padding: 14px;
            border-radius: 8px;
            font-size: 16px;
            line-height: 1.5;
        }

        /* Documentos */
        .docs-grid {
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 14px;
            margin-top: 14px;
        }
        .doc-box {
            background: white;
            padding: 12px;
            border-radius: 6px;
            border-left: 4px solid #1C7C54;
            font-size: 14px;
        }

        /* Sugerencias */
        .suggestion-box {
            background: #E8F3EE;
            padding: 10px;
            border-radius: 6px;
            margin-bottom: 12px;
            border-left: 4px solid #60A68C;
        }
        </style>

        <div class='navbar'>Asistente Normativo MINAM Perú</div>
        <div class='banner'>
            <img src="https://www.gob.pe/institucion/minam">
        </div>
    """, unsafe_allow_html=True)

    # Ocultar sidebar
    with st.sidebar:
        st.empty()

    st.title("🌱 Consulta Normativa Ambiental")
    st.write("Asistente legal contextualizado a la normativa ambiental oficial del MINAM.")

    # Sugerencias
    st.markdown("""
        <div class='suggestion-box'>
        <b>Ejemplos de consultas:</b>
        <ul>
            <li>¿Cuál es el procedimiento para obtener autorización ambiental?</li>
            <li>¿Qué normativa regula la gestión de residuos peligrosos?</li>
            <li>Requisitos de evaluación de impacto ambiental para proyectos mineros.</li>
        </ul>
        </div>
    """, unsafe_allow_html=True)

    vstore = load_and_process_documents()
    if not vstore:
        st.stop()

    if not GEMINI_API_KEY:
        st.error("Falta GEMINI_API_KEY en archivo .env")
        st.stop()

    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.0, google_api_key=GEMINI_API_KEY)
    retriever = vstore.as_retriever(search_kwargs={"k": 5})

    prompt = ChatPromptTemplate.from_template("""
        Responde como asesor jurídico del MINAM Perú.
        Responde solo con fundamentos normativos presentes en el contexto.
        Sé claro, técnico y exacto.

        CONTEXTO:
        {context}

        PREGUNTA:
        {question}
    """)

    def format_docs(docs):
        return "\n\n".join(x.page_content for x in docs)

    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt | llm | StrOutputParser()
    )

    # Formulario de consulta
    with st.form("consulta"):
        consulta = st.text_input("📌 Escribe tu consulta", key="user_query_input")
        submit = st.form_submit_button("Consultar")
        limpiar = st.form_submit_button("Limpiar", on_click=clear_query)

    if submit and consulta:
        with st.spinner("Procesando consulta normativa..."):
            respuesta = chain.invoke(consulta)
            fuentes = retriever.invoke(consulta)

        st.subheader("✅ Respuesta")
        st.markdown(f"<div class='response-box'>{respuesta}</div>", unsafe_allow_html=True)

        st.subheader("📄 Documentos consultados")
        st.markdown("<div class='docs-grid'>", unsafe_allow_html=True)

        for d in fuentes:
            titulo = d.metadata.get("titulo", "Documento")
            st.markdown(f"""
                <div class='doc-box'>
                <b>{titulo}</b><br>
                <small>{d.page_content[:220]}...</small>
                </div>
            """, unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
