# llm_service.py
import os
import re
from typing import Dict, List, Optional, Tuple
import hashlib

class LLMService:
    """
    Servicio para interactuar con LLMs externos (OpenAI, Anthropic, Gemini)
    Separado de la lógica principal del RAG
    """

    def __init__(self, api_key: Optional[str] = None, proveedor: str = "openai",
                 usar_fallback: bool = True):
        """
        Inicializa el servicio de LLM

        Args:
            api_key: API key del proveedor (si None, busca en variables de entorno)
            proveedor: "openai", "anthropic", o "gemini"
            usar_fallback: Si True, usa método simple cuando falla el LLM
        """
        self.proveedor = proveedor
        self.usar_fallback = usar_fallback
        self.api_key = api_key or self._obtener_api_key_env()
        self.cliente = None
        self._inicializar_cliente()

    def _obtener_api_key_env(self) -> Optional[str]:
        """Obtiene API key de variables de entorno"""
        env_vars = {
            "openai": "OPENAI_API_KEY",
            "anthropic": "ANTHROPIC_API_KEY",
            "gemini": "GEMINI_API_KEY"
        }
        var_name = env_vars.get(self.proveedor)
        return os.getenv(var_name) if var_name else None

    def _inicializar_cliente(self):
        """Inicializa el cliente según el proveedor"""
        if not self.api_key:
            print(f"⚠️ No hay API key para {self.proveedor}. Usando modo fallback.")
            return

        try:
            if self.proveedor == "openai":
                import openai
                self.cliente = openai.OpenAI(api_key=self.api_key)
            elif self.proveedor == "anthropic":
                import anthropic
                self.cliente = anthropic.Anthropic(api_key=self.api_key)
            elif self.proveedor == "gemini":
                import google.generativeai as genai
                genai.configure(api_key=self.api_key)
                self.cliente = genai
        except ImportError as e:
            print(f"❌ Error importando librería para {self.proveedor}: {e}")
            self.cliente = None

    def generar_respuesta(self, pregunta: str, contexto: str,
                          metadatos: Optional[Dict] = None,
                          temperatura: float = 0.3) -> str:
        """
        Genera respuesta usando LLM externo o fallback

        Args:
            pregunta: Pregunta del usuario
            contexto: Contexto recuperado del RAG
            metadatos: Metadatos del documento (páginas, etc.)
            temperatura: Creatividad de la respuesta (0-1)

        Returns:
            Respuesta generada
        """
        # Si no hay cliente válido, usar fallback
        if not self.cliente or not self.api_key:
            return self._respuesta_fallback(pregunta, contexto, metadatos)

        try:
            # Extraer páginas de metadatos
            paginas = self._extraer_paginas(metadatos)

            # Construir prompt
            prompt = self._construir_prompt(pregunta, contexto, paginas)

            # Llamar al LLM según proveedor
            if self.proveedor == "openai":
                respuesta = self._llamada_openai(prompt, temperatura)
            elif self.proveedor == "anthropic":
                respuesta = self._llamada_anthropic(prompt, temperatura)
            elif self.proveedor == "gemini":
                respuesta = self._llamada_gemini(prompt, temperatura)
            else:
                respuesta = self._respuesta_fallback(pregunta, contexto, metadatos)

            # Añadir info de páginas si no está incluida
            respuesta = self._agregar_info_paginas(respuesta, paginas)

            return respuesta

        except Exception as e:
            print(f"❌ Error con LLM {self.proveedor}: {e}")
            if self.usar_fallback:
                return self._respuesta_fallback(pregunta, contexto, metadatos)
            return f"Error generando respuesta: {e}"

    def _construir_prompt(self, pregunta: str, contexto: str, paginas: List[int]) -> str:
        """Construye el prompt para el LLM"""
        texto_paginas = f"páginas {', '.join(map(str, paginas))}" if paginas else "documento"

        prompt = f"""Eres un asistente legal especializado en análisis de documentos laborales.

Pregunta del usuario: {pregunta}

Contexto del {texto_paginas}:
{contexto[:2500]}

Instrucciones:
1. Responde ÚNICAMENTE basado en el contexto proporcionado
2. Si la información no está en el contexto, indica claramente "No se encuentra en el documento"
3. Sé conciso, profesional y específico
4. Menciona los montos, fechas o datos numéricos exactos cuando aparezcan
5. No inventes información que no esté en el contexto

Respuesta:"""

        return prompt

    def _llamada_openai(self, prompt: str, temperatura: float) -> str:
        """Llama a OpenAI GPT"""
        respuesta = self.cliente.chat.completions.create(
            model="gpt-3.5-turbo",  # o "gpt-4"
            messages=[
                {"role": "system", "content": "Eres un asistente legal experto."},
                {"role": "user", "content": prompt}
            ],
            temperature=temperatura,
            max_tokens=500
        )
        return respuesta.choices[0].message.content

    def _llamada_anthropic(self, prompt: str, temperatura: float) -> str:
        """Llama a Anthropic Claude"""
        respuesta = self.cliente.messages.create(
            model="claude-3-haiku-20240307",
            max_tokens=500,
            temperature=temperatura,
            messages=[{"role": "user", "content": prompt}]
        )
        return respuesta.content[0].text

    def _llamada_gemini(self, prompt: str, temperatura: float) -> str:
        """Llama a Google Gemini"""
        model = self.cliente.GenerativeModel('gemini-pro')
        respuesta = model.generate_content(
            prompt,
            generation_config={"temperature": temperatura, "max_output_tokens": 500}
        )
        return respuesta.text

    def _respuesta_fallback(self, pregunta: str, contexto: str,
                            metadatos: Optional[Dict] = None) -> str:
        """
        Método de fallback basado en reglas (similar a tu generar_respuesta_simple)
        """
        pregunta_lower = pregunta.lower()

        # Indemnización
        if "indemnización" in pregunta_lower or "indemnizacion" in pregunta_lower:
            if "$" in contexto or "indemnización" in contexto:
                respuesta = "Según el documento, se trata de una indemnización."
                numeros = re.findall(r'\$\d+(?:,\d+)?', contexto)
                if numeros:
                    respuesta += f" Montos encontrados: {', '.join(numeros)}"
                return respuesta

        # Despido
        elif "despido" in pregunta_lower:
            if "improcedente" in contexto:
                return "El documento indica que el despido fue declarado IMPROCEDENTE."
            elif "procedente" in contexto:
                return "El documento indica que el despido fue declarado PROCEDENTE."

        # Páginas
        if "página" in pregunta_lower or "pagina" in pregunta_lower:
            if metadatos:
                paginas = self._extraer_paginas(metadatos)
                if paginas:
                    return f"La información relevante se encuentra en las páginas: {', '.join(map(str, paginas))}"

        # Respuesta genérica
        return f"Según el documento: {contexto[:300]}..."

    def _extraer_paginas(self, metadatos: Optional[Dict]) -> List[int]:
        """Extrae números de página de los metadatos"""
        if not metadatos:
            return []

        paginas = []
        if 'metadatos' in metadatos:
            for m in metadatos['metadatos'][0]:
                if 'pagina' in m:
                    paginas.append(m['pagina'])
        elif 'pagina' in metadatos:
            paginas = [metadatos['pagina']] if isinstance(metadatos['pagina'], int) else metadatos['pagina']

        return list(set(paginas))  # Eliminar duplicados

    def _agregar_info_paginas(self, respuesta: str, paginas: List[int]) -> str:
        """Añade información de páginas si no está presente"""
        if paginas and "página" not in respuesta.lower() and "pagina" not in respuesta.lower():
            respuesta += f"\n\n📄 Información encontrada en páginas: {', '.join(map(str, paginas))}"
        return respuesta

    def limpiar_cache(self):
        """Limpia caché interna si se implementa"""
        pass


# Función helper para crear instancia fácilmente
def crear_llm_service(proveedor: str = "openai", api_key: Optional[str] = None) -> LLMService:
    """Factory function para crear servicio LLM"""
    return LLMService(api_key=api_key, proveedor=proveedor)