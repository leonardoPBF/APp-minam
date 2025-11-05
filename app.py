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

# Cargar variables
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Cargar documentos
@st.cache_resource
def load_and_process_documents():
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

def clear_query():
    st.session_state["user_query_input"] = ""
    st.rerun()

def main():
    st.set_page_config(page_title="Asistente Normativo MINAM", layout="wide")

    # ===================== CSS =====================
    st.markdown("""
        <style>
        body { background-color: #F4F8F7; }
        .stApp { background: #F4F8F7; color: #1A3D34; }

        .navbar {
            background-color: #D8EEE4;
            width: 100%;
            padding: 16px;
            color: #0E3B2E;
            font-size: 22px;
            text-align: center;
            border-bottom: 3px solid #9DC8B9;
            font-weight: 600;
            border-radius: 6px;
            margin-bottom: 10px;
        }

        .banner img {
            width: 100%;
            height: 160px;
            object-fit: cover;
            border-radius: 6px;
            margin-bottom: 15px;
        }

        .response-box {
            background: #FFFFFF;
            border-left: 4px solid #4CA476;
            padding: 16px;
            border-radius: 8px;
            font-size: 16px;
        }

        .docs-grid {
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 14px;
            margin-top: 14px;
        }

        .doc-box {
            background: #FFFFFF;
            padding: 12px;
            border-radius: 6px;
            border-left: 4px solid #4CA476;
            font-size: 14px;
        }

        .suggestion-box {
            background: #EDF7F2;
            padding: 10px;
            border-radius: 6px;
            margin-bottom: 12px;
            border-left: 4px solid #60A68C;
        }

        textarea, input {
            background-color: #ffffff !important;
            border-radius: 8px !important;
            border: 1px solid #C8E2D1 !important;
        }

        /* Ocultar botones internos del form */
        form button {
            display: none !important;
        }
        
        textarea, input {
            background-color: #ffffff !important;
            border-radius: 8px !important;
            border: 1px solid #C8E2D1 !important;
            color: #1A3D34 !important; /* Color de texto visible */
        }

        textarea::placeholder, input::placeholder {
            color: #6B8F88 !important; /* Color suave para placeholder */
        }

        button{
          background-color: #1A3D34 !important; /* Color de texto visible */  
          color: #6B8F88 !important; 
        }

        <div class='navbar'>Asistente Normativo MINAM Perú</div>
        <div class='banner'>
            <img src="https://www.conservamospornaturaleza.org/img/2015/10/minam-logo.jpg">
        </div>
    """, unsafe_allow_html=True)

    # ===================== UI =====================
    st.title("🌱 Consulta Normativa Ambiental")
    st.write("Asistente contextualizado a la normativa ambiental oficial del MINAM Perú.")

    st.markdown("""
        <div class='suggestion-box'>
        <b>Ejemplos de preguntas:</b>
        <ul>
            <li>¿Qué norma regula la gestión de residuos sólidos?</li>
            <li>Requisitos para una evaluación de impacto ambiental (EIA).</li>
            <li>¿Cómo tramitar una autorización ambiental?</li>
        </ul>
        </div>
    """, unsafe_allow_html=True)

    vstore = load_and_process_documents()
    if not GEMINI_API_KEY:
        st.error("❌ Falta GEMINI_API_KEY en el archivo .env")
        st.stop()

    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.0, google_api_key=GEMINI_API_KEY)
    retriever = vstore.as_retriever(search_kwargs={"k": 5})

    prompt = ChatPromptTemplate.from_template("""
        Responde como asesor jurídico del MINAM Perú.
        Usa únicamente fundamentos normativos presentes en el contexto.

        CONTEXTO:
        {context}

        PREGUNTA:
        {question}
    """)

    chain = (
        {"context": retriever | (lambda docs: "\n\n".join(x.page_content for x in docs)),
         "question": RunnablePassthrough()}
        | prompt | llm | StrOutputParser()
    )

    # ===================== FORM =====================
    with st.form("consulta"):
        consulta = st.text_input("✏️ Escribe tu consulta", key="user_query_input")
        submit = st.form_submit_button("hidden_submit")

    col1, col2 = st.columns(2)

    # Botón consultar
    with col1:
        if st.button("✅ Consultar", use_container_width=True):
            submit = True

    # Botón limpiar
    with col2:
        if st.button("🧹 Limpiar", use_container_width=True):
            clear_query()

    # ===================== RESPUESTA =====================
    if submit and consulta:
        with st.spinner("Procesando normativa..."):
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
