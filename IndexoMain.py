# ============================================
# RAG LEGAL - Asistente para casos
# ============================================

import os
from pypdf import PdfReader
import chromadb
from chromadb.utils import embedding_functions
from sentence_transformers import SentenceTransformer
import hashlib

class AsistenteLegalRAG:
    """
    Sistema RAG para abogados: sube un PDF, hazle preguntas.
    """

    def __init__(self, nombre_caso):
        """
        Inicializa el asistente para un caso específico
        """
        self.nombre_caso = nombre_caso

        # 1. Configurar ChromaDB (persistente en disco)
        self.chroma_client = chromadb.PersistentClient(path=f"./chroma_db_{nombre_caso}")

        # 2. Configurar embedding (local, gratis)
        self.embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )

        # 3. Crear o cargar colección
        try:
            self.coleccion = self.chroma_client.get_collection(
                name=nombre_caso,
                embedding_function=self.embedding_fn
            )
            print(f"✅ Colección '{nombre_caso}' cargada (ya existía)")
        except:
            self.coleccion = self.chroma_client.create_collection(
                name=nombre_caso,
                embedding_function=self.embedding_fn
            )
            print(f"✅ Colección '{nombre_caso}' creada (nueva)")

    def extraer_texto_pdf(self, ruta_pdf):
        """
        Extrae texto de un PDF página por página
        """
        print(f"📄 Leyendo PDF: {ruta_pdf}")
        reader = PdfReader(ruta_pdf)
        paginas_texto = []

        for i, pagina in enumerate(reader.pages):
            texto = pagina.extract_text()
            if texto.strip():
                paginas_texto.append({
                    "numero": i + 1,
                    "texto": texto
                })

        print(f"   ✅ Extraídas {len(paginas_texto)} páginas")
        return paginas_texto

    def chunking_legal(self, paginas_texto, chunk_size=800, overlap=100):
        """
        Divide el texto en chunks inteligentes
        """
        chunks = []

        for pagina in paginas_texto:
            texto = pagina["texto"]

            # Dividir por párrafos primero
            parrafos = texto.split('\n\n')

            for parrafo in parrafos:
                if len(parrafo) < 50:
                    continue

                # Si el párrafo es muy largo, dividirlo
                if len(parrafo) > chunk_size:
                    for i in range(0, len(parrafo), chunk_size - overlap):
                        chunk_texto = parrafo[i:i + chunk_size]
                        chunks.append({
                            "texto": chunk_texto,
                            "pagina": pagina["numero"],
                            "chunk_id": f"pag{pagina['numero']}_{len(chunks)}"
                        })
                else:
                    chunks.append({
                        "texto": parrafo,
                        "pagina": pagina["numero"],
                        "chunk_id": f"pag{pagina['numero']}_{len(chunks)}"
                    })

        print(f"   ✅ Creados {len(chunks)} chunks legales")
        return chunks

    def indexar_caso(self, ruta_pdf):
        """
        Indexa el PDF en la base vectorial (solo se hace UNA VEZ)
        """
        print(f"\n📚 INDEXANDO CASO: {self.nombre_caso}")
        print("=" * 50)

        # 1. Extraer texto
        paginas = self.extraer_texto_pdf(ruta_pdf)

        # 2. Crear chunks
        chunks = self.chunking_legal(paginas)

        # 3. Limpiar colección existente (opcional)
        try:
            self.coleccion.delete()
        except:
            pass

        # 4. Añadir a ChromaDB
        ids = [chunk["chunk_id"] for chunk in chunks]
        textos = [chunk["texto"] for chunk in chunks]
        metadatos = [
            {"pagina": chunk["pagina"], "texto_corto": chunk["texto"][:100]}
            for chunk in chunks
        ]

        self.coleccion.add(
            ids=ids,
            documents=textos,
            metadatas=metadatos
        )

        print(f"\n✅ CASO INDEXADO CORRECTAMENTE")
        print(f"   Total chunks: {len(chunks)}")
        print(f"   Puedes hacer preguntas ahora.")

        return chunks

    def preguntar(self, pregunta, top_k=3):
        """
        Hace una pregunta sobre el caso
        """
        print(f"\n🔍 PREGUNTA: {pregunta}")
        print("-" * 50)

        # 1. Buscar chunks relevantes
        resultados = self.coleccion.query(
            query_texts=[pregunta],
            n_results=top_k
        )

        if not resultados['documents'][0]:
            print("❌ No se encontraron fragmentos relevantes")
            return None

        # 2. Mostrar fragmentos encontrados
        print(f"\n📖 Fragmentos relevantes encontrados:")
        for i, (texto, metadata) in enumerate(zip(
                resultados['documents'][0],
                resultados['metadatas'][0]
        )):
            print(f"\n   [{i+1}] Página {metadata['pagina']}:")
            print(f"   {texto[:300]}...")

        # 3. Construir contexto
        contexto = "\n\n---\n\n".join(resultados['documents'][0])

        # 4. Simular respuesta del LLM (por ahora)
        # En un RAG real, aquí llamarías a OpenAI/DeepSeek
        print(f"\n" + "=" * 50)
        print("📝 RESPUESTA (según los documentos):")
        print("-" * 50)

        # Respuesta basada en el contexto encontrado
        respuesta = self.generar_respuesta_simple(pregunta, contexto, resultados)

        print(respuesta)

        return {
            "pregunta": pregunta,
            "respuesta": respuesta,
            "fuentes": [{"pagina": m["pagina"]} for m in resultados['metadatas'][0]]
        }

    def generar_respuesta_simple(self, pregunta, contexto, resultados):
        """
        Genera respuesta básica (sin LLM externo)
        Para probar el RAG sin API key
        """
        # Buscar palabras clave en el contexto
        respuesta = ""

        if "indemnización" in pregunta.lower() or "indemnizacion" in pregunta.lower():
            if "$" in contexto or "indemnización" in contexto:
                respuesta = "Según el documento, se trata de una indemnización."
                # Buscar números
                import re
                numeros = re.findall(r'\$\d+(?:,\d+)?', contexto)
                if numeros:
                    respuesta += f" Montos encontrados: {', '.join(numeros)}"

        elif "despido" in pregunta.lower():
            if "improcedente" in contexto:
                respuesta = "El documento indica que el despido fue declarado IMPROCEDENTE."
            elif "procedente" in contexto:
                respuesta = "El documento indica que el despido fue declarado PROCEDENTE."
            else:
                respuesta = "Revisa el documento para determinar la naturaleza del despido."

        elif "página" in pregunta.lower() or "pagina" in pregunta.lower():
            paginas = [m["pagina"] for m in resultados['metadatas'][0]]
            respuesta = f"La información relevante se encuentra en las páginas: {', '.join(map(str, paginas))}"

        else:
            respuesta = f"Según el documento: {contexto[:500]}..."

        return respuesta

    def chat_interactivo(self):
        """
        Modo chat para hacer múltiples preguntas
        """
        print("\n" + "=" * 60)
        print(f"⚖️  ASISTENTE LEGAL RAG - Caso: {self.nombre_caso}")
        print("=" * 60)
        print("\nEscribe 'salir' para terminar")
        print("Escribe 'fuentes' para ver los fragmentos encontrados")
        print("-" * 60)

        while True:
            pregunta = input("\n❓ Tu pregunta: ").strip()

            if pregunta.lower() == 'salir':
                print("\n👋 ¡Hasta luego!")
                break

            if pregunta.lower() == 'fuentes':
                print("\n📚 Las fuentes son los PDFs que indexaste")
                continue

            if not pregunta:
                continue

            resultado = self.preguntar(pregunta)

            if resultado and resultado.get("fuentes"):
                print(f"\n📌 Fuentes: Páginas {', '.join(str(f['pagina']) for f in resultado['fuentes'])}")


# ============================================
# EJECUCIÓN PRINCIPAL
# ============================================

if __name__ == "__main__":
    print("⚖️  SISTEMA RAG PARA CASOS LEGALES")
    print("=" * 50)

    # Configuración
    NOMBRE_CASO = "caso_permiso_1"
    RUTA_PDF = "C:\\Users\\mitno\\Documents\\EXPEDIENTE_1.pdf"  # PDF path

    # Crear asistente
    asistente = AsistenteLegalRAG(NOMBRE_CASO)

    # Preguntar si indexar o no
    if input("¿Indexar nuevo PDF? (s/n): ").lower() == 's':
        asistente.indexar_caso(RUTA_PDF)

    # Iniciar chat
    asistente.chat_interactivo()