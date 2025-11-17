"""
VaccinIA v3.3 - Communication Module - API Backend con RAG
NUEVO: Sistema de búsqueda especializada por condición médica
- Embarazo, VIH, Cáncer, Trasplantes, Asplenia, Diabetes, EPOC, ERC, Adultos Mayores
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any
import json
from datetime import datetime
import os
import re

# RAG Stack
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document

# ============================================================================
# ESTRATEGIAS DE BÚSQUEDA POR CONDICIÓN
# ============================================================================

CONDITION_SEARCH_STRATEGIES = {
    "embarazo": {
        "critical_vaccines": ["Tdap", "COVID-19", "Influenza", "RSV"],
        "k_docs": 10,
        "forced_queries": [
            "Tdap tosferina embarazo semana 27 36",
            "COVID-19 embarazo cualquier trimestre",
            "Influenza embarazo preferencia 20 semanas",
            "RSV VRS embarazo 32 semanas nirsevimab protección neonatal"
        ],
        "description": "Gestantes - cualquier trimestre"
    },
    
    "adulto_mayor": {
        "critical_vaccines": [
            "Neumococo PCV13", "Neumococo PPSV23",
            "Herpes Zóster", "Tdap", 
            "Influenza", "COVID-19"
        ],
        "k_docs": 20,
        "forced_queries": [
            "Neumococo conjugada adultos 65 años",
            "Neumococo polisacárida adultos 65 años",
            "Herpes Zóster indicaciones edad 50 años 65",
            "Herpes Zóster Shingrix 2 dosis",
            "Tdap refuerzo adultos",
            "Influenza anual adultos",
            "COVID-19 adultos mayores"
        ],
        "description": "Adultos ≥65 años"
    },
    
    "vih": {
        "critical_vaccines": [
            "Neumococo PCV13", "Neumococo PPSV23",
            "Hepatitis B", "Tdap", 
            "Meningococo ACYW", "Meningococo B (Bexsero)",
            "VPH", "COVID-19", "RSV"
        ],
        "k_docs": 17,
        "force_by_metadata": ["Meningococo B (Bexsero)"],
        "forced_queries": [
            "Neumococo PCV13 VIH cualquier CD4",
            "Neumococo PPSV23 VIH después PCV13",
            "Hepatitis B VIH CD4",
            "Tdap VIH sin esquema previo",
            "Meningococo ACYW VIH CD4",
            "VPH VIH 3 dosis hasta 26 años"
        ],
        "description": "Pacientes con VIH/SIDA"
    },
    
    "cancer": {
        "critical_vaccines": [
            "Neumococo PCV13", "Neumococo PPSV23",
            "Influenza", "COVID-19", "RSV"
        ],
        "k_docs": 15,
        "contraindicated": ["vacunas vivas", "MMR", "Varicela", "Fiebre Amarilla", "Herpes Zóster vivo"],
        "forced_queries": [
            "Neumococo cáncer quimioterapia inmunosupresión",
            "Influenza cáncer durante tratamiento activo",
            "COVID-19 cáncer inmunosupresión",
            "contraindicación vacunas vivas cáncer quimioterapia",
            "timing vacunación antes después quimioterapia"
        ],
        "timing_note": "Idealmente 2 semanas antes o 3 meses después de quimioterapia",
        "description": "Pacientes oncológicos en quimioterapia/radioterapia"
    },
    
    "trasplante_organo": {
        "critical_vaccines": [
            "Neumococo PCV13", "Neumococo PPSV23",
            "Influenza", "COVID-19", "RSV"
        ],
        "k_docs": 15,
        "contraindicated": ["vacunas vivas"],
        "forced_queries": [
            "Neumococo trasplante órgano sólido",
            "Influenza trasplante inmunosupresión",
            "COVID-19 trasplante receptor",
            "contraindicación vacunas vivas trasplante"
        ],
        "description": "Receptores de trasplante de órgano sólido"
    },
    
    "trasplante_medula": {
        "critical_vaccines": [
            "Neumococo PCV13", "Neumococo PPSV23",
            "Influenza", "COVID-19"
        ],
        "k_docs": 15,
        "contraindicated": ["vacunas vivas"],
        "forced_queries": [
            "trasplante médula ósea stem cell HSCT",
            "Neumococo PCV13 PPSV23 trasplante hematopoyético",
            "vacunación después trasplante médula 6 meses",
            "contraindicación vacunas vivas trasplante médula"
        ],
        "description": "Receptores de trasplante de médula ósea/stem cell"
    },
    
    "asplenia": {
        "critical_vaccines": [
            "Neumococo PCV13", "Neumococo PPSV23",
            "Meningococo ACYW", "Meningococo B (Bexsero)",
            "Haemophilus influenzae tipo b"
        ],
        "k_docs": 12,
        "force_by_metadata": ["Meningococo B (Bexsero)"],
        "forced_queries": [
            "Neumococo PCV13 PPSV23 asplenia esplenectomía",
            "Meningococo ACYW asplenia 2 dosis",
            "Meningococo B asplenia",
            "Haemophilus influenzae asplenia",
            "vacunación antes esplenectomía 2 semanas"
        ],
        "description": "Asplenia anatómica o funcional"
    },
    
    "diabetes": {
        "critical_vaccines": [
            "Neumococo PCV13", "Neumococo PPSV23",
            "Influenza", "COVID-19", "Hepatitis B"
        ],
        "k_docs": 12,
        "forced_queries": [
            "Neumococo diabetes mellitus",
            "Hepatitis B diabetes <60 años",
            "Influenza diabetes anual",
            "COVID-19 diabetes"
        ],
        "description": "Diabetes mellitus tipo 1 o 2"
    },
    
    "epoc": {
        "critical_vaccines": [
            "Neumococo PCV13", "Neumococo PPSV23",
            "Influenza", "COVID-19"
        ],
        "k_docs": 10,
        "forced_queries": [
            "Neumococo EPOC enfermedad pulmonar",
            "Influenza EPOC anual",
            "COVID-19 EPOC"
        ],
        "description": "Enfermedad Pulmonar Obstructiva Crónica"
    },
    
    "erc": {
        "critical_vaccines": [
            "Neumococo PCV13", "Neumococo PPSV23",
            "Hepatitis B", "Influenza", "COVID-19"
        ],
        "k_docs": 12,
        "forced_queries": [
            "Neumococo enfermedad renal crónica",
            "Hepatitis B hemodiálisis diálisis",
            "Influenza enfermedad renal",
            "COVID-19 insuficiencia renal"
        ],
        "description": "Enfermedad Renal Crónica"
    },
    
    "hepatopatia": {
        "critical_vaccines": [
            "Neumococo PCV13", "Neumococo PPSV23",
            "Hepatitis A", "Hepatitis B",
            "Influenza", "COVID-19"
        ],
        "k_docs": 12,
        "forced_queries": [
            "Neumococo cirrosis hepatopatía",
            "Hepatitis A B enfermedad hepática",
            "vacunación enfermedad hígado crónica"
        ],
        "description": "Enfermedad hepática crónica"
    },
    
    "inmunosupresion": {
        "critical_vaccines": [
            "Neumococo PCV13", "Neumococo PPSV23",
            "Influenza", "COVID-19"
        ],
        "k_docs": 12,
        "contraindicated": ["vacunas vivas"],
        "forced_queries": [
            "Neumococo inmunosupresión biológicos",
            "vacunación inmunosupresor anti-TNF rituximab",
            "contraindicación vacunas vivas inmunosupresión",
            "timing vacunación biológicos antes después"
        ],
        "description": "Terapia inmunosupresora (biológicos, esteroides)"
    },
    
    "enfermedad_autoinmune": {
        "critical_vaccines": [
            "Neumococo PCV13", "Neumococo PPSV23",
            "Influenza", "COVID-19"
        ],
        "k_docs": 10,
        "forced_queries": [
            "Neumococo lupus artritis reumatoide autoinmune",
            "vacunación enfermedad reumatológica",
            "Influenza lupus artritis"
        ],
        "description": "Enfermedades autoinmunes (LES, AR, etc.)"
    }
}

# Mapeo de keywords a condiciones
CONDITION_KEYWORDS = {
    "embarazo": ['embaraz', 'gestante', 'gestación', 'prenatal'],
    "adulto_mayor": ['adulto mayor', 'adulta mayor', 'tercera edad', 'anciano', 'geriatría', 'geriátrico'],
    "vih": ['vih', 'sida', 'cd4', 'hiv'],
    "cancer": ['cáncer', 'cancer', 'quimioterapia', 'radioterapia', 'oncológico', 
               'oncologia', 'tumor', 'neoplasia', 'leucemia', 'linfoma'],
    "trasplante_organo": ['trasplante órgano', 'trasplante renal', 'trasplante hepático', 
                          'trasplante corazón', 'receptor órgano'],
    "trasplante_medula": ['trasplante médula', 'trasplante stem cell', 'hsct', 
                          'trasplante hematopoyético'],
    "asplenia": ['asplenia', 'hiposplenia', 'esplenectomía', 'sin bazo'],
    "diabetes": ['diabet', 'diabético'],
    "epoc": ['epoc', 'copd', 'enfermedad pulmonar obstructiva'],
    "erc": ['renal crónica', 'insuficiencia renal', 'hemodiálisis', 'diálisis', 'erc'],
    "hepatopatia": ['cirrosis', 'hepatopatía', 'hepática crónica', 'enfermedad hígado'],
    "inmunosupresion": ['inmunosupres', 'inmunocomprometi', 'inmunodeprimi',
                        'biológico', 'anti-tnf', 'rituximab', 'metotrexate',
                        'corticoide', 'esteroide', 'prednisona', 'azatioprina'],
    "enfermedad_autoinmune": ['lupus', 'artritis reumatoide', 'autoinmune', 
                              'enfermedad reumatológica']
}

# ============================================================================
# MODELOS PYDANTIC
# ============================================================================

class PatientProfile(BaseModel):
    age: int = Field(..., ge=18, le=120)
    sex: str = Field(..., pattern="^(M|F)$")
    pregnant: Optional[bool] = Field(False)
    immunocompromised: Optional[bool] = Field(False)
    chronic_conditions: Optional[str] = Field(None)
    occupation: Optional[str] = Field(None)
    travel_history: Optional[List[str]] = Field(default_factory=list)
    vaccination_history: Optional[Dict[str, Any]] = Field(default_factory=dict)

class VaccinationQuery(BaseModel):
    question: str = Field(..., min_length=10)
    patient_profile: Optional[PatientProfile] = None

class SourceInfo(BaseModel):
    vaccine: str
    section: str
    content_preview: str
    source_file: str

class ChatResponse(BaseModel):
    answer: str
    confidence: str
    sources: List[SourceInfo]
    recommendations: Optional[List[str]] = None
    timestamp: str

# ============================================================================
# INICIALIZACIÓN FASTAPI
# ============================================================================

app = FastAPI(
    title="VaccinIA v3.3 Communication Module API",
    description="Sistema inteligente de recomendaciones de vacunación con búsqueda especializada",
    version="3.3.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================================
# RAG SYSTEM
# ============================================================================

class VaccineRAGSystem:
    def __init__(self):
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        if not self.openai_api_key:
            raise ValueError("OPENAI_API_KEY no configurada")
        
        self.embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small",
            openai_api_key=self.openai_api_key
        )
        
        self.vectorstore = Chroma(
            persist_directory="./chroma_vaccinia",
            embedding_function=self.embeddings
        )
        
        self.llm = ChatOpenAI(
            model="gpt-4-turbo-preview",
            temperature=0,
            openai_api_key=self.openai_api_key
        )
    
    def detect_age_from_text(self, text: str) -> Optional[int]:
        """
        Detecta edad numérica en el texto (ej: '76 años', '68 años')
        """
        # Patrón para detectar números seguidos de 'años' o 'año'
        age_pattern = r'\b(\d{1,3})\s*(?:años?|a[ñn]os?)\b'
        matches = re.findall(age_pattern, text.lower())
        
        if matches:
            try:
                age = int(matches[0])
                # Validar que sea una edad razonable
                if 18 <= age <= 120:
                    return age
            except ValueError:
                pass
        
        return None
    
    def detect_conditions(self, question: str, patient_profile: Optional[PatientProfile] = None) -> List[str]:
        """
        Detecta condiciones médicas en la pregunta o perfil del paciente
        INCLUYE detección automática de edad para adultos mayores
        """
        question_lower = question.lower()
        conditions = []
        
        # NUEVO: Detectar edad numérica para adulto mayor
        detected_age = self.detect_age_from_text(question)
        if detected_age and detected_age >= 65:
            if 'adulto_mayor' not in conditions:
                conditions.append('adulto_mayor')
        
        # Si hay patient_profile con edad
        if patient_profile and patient_profile.age >= 65:
            if 'adulto_mayor' not in conditions:
                conditions.append('adulto_mayor')
        
        # Iterar sobre todas las condiciones y sus keywords
        for condition, keywords in CONDITION_KEYWORDS.items():
            for keyword in keywords:
                if keyword in question_lower:
                    conditions.append(condition)
                    break
        
        # Verificar también patient_profile si existe
        if patient_profile:
            if patient_profile.pregnant and 'embarazo' not in conditions:
                conditions.append('embarazo')
            
            if patient_profile.immunocompromised:
                specific_conditions = {'cancer', 'vih', 'trasplante_organo', 'trasplante_medula'}
                if not any(c in conditions for c in specific_conditions):
                    if 'inmunosupresion' not in conditions:
                        conditions.append('inmunosupresion')
            
            # Analizar chronic_conditions del perfil
            if patient_profile.chronic_conditions:
                chronic_lower = patient_profile.chronic_conditions.lower()
                
                for condition, keywords in CONDITION_KEYWORDS.items():
                    if condition not in conditions:
                        for keyword in keywords:
                            if keyword in chronic_lower:
                                conditions.append(condition)
                                break
        
        return list(set(conditions))  # Eliminar duplicados
    
    def retrieve_condition_docs(self, condition: str, question: str) -> List[Document]:
        """
        Recupera documentos específicos para una condición médica usando estrategia especializada
        """
        strategy = CONDITION_SEARCH_STRATEGIES.get(condition)
        if not strategy:
            return []
        
        docs = []
        
        # 1. Búsquedas forzadas específicas
        for forced_query in strategy.get("forced_queries", []):
            results = self.vectorstore.similarity_search(forced_query, k=2)
            docs.extend(results)
        
        # 2. Si hay force_by_metadata, recuperar por metadata
        for vaccine_name in strategy.get("force_by_metadata", []):
            try:
                forced_docs = self.vectorstore.get(
                    where={"vaccine": vaccine_name},
                    limit=2
                )
                if forced_docs and 'documents' in forced_docs:
                    for i, doc_content in enumerate(forced_docs['documents']):
                        metadata = forced_docs['metadatas'][i] if 'metadatas' in forced_docs else {}
                        docs.append(Document(page_content=doc_content, metadata=metadata))
            except:
                pass
        
        # 3. Búsqueda semántica general
        general_results = self.vectorstore.similarity_search(
            f"{condition} {question}",
            k=strategy.get("k_docs", 10)
        )
        docs.extend(general_results)
        
        # Eliminar duplicados por contenido
        seen_contents = set()
        unique_docs = []
        for doc in docs:
            if doc.page_content not in seen_contents:
                seen_contents.add(doc.page_content)
                unique_docs.append(doc)
        
        return unique_docs[:strategy.get("k_docs", 15)]
    
    def answer_question(self, question: str, patient_profile: Optional[PatientProfile] = None) -> ChatResponse:
        """
        Responde pregunta usando RAG con búsqueda especializada por condición
        """
        # Detectar condiciones
        detected_conditions = self.detect_conditions(question, patient_profile)
        
        # Recuperar documentos
        if detected_conditions:
            all_docs = []
            for condition in detected_conditions:
                condition_docs = self.retrieve_condition_docs(condition, question)
                all_docs.extend(condition_docs)
            
            # Eliminar duplicados finales
            seen = set()
            relevant_docs = []
            for doc in all_docs:
                if doc.page_content not in seen:
                    seen.add(doc.page_content)
                    relevant_docs.append(doc)
        else:
            # Búsqueda general si no se detectan condiciones
            relevant_docs = self.vectorstore.similarity_search(question, k=10)
        
        # Construir contexto
        context = "\n\n---\n\n".join([doc.page_content for doc in relevant_docs[:15]])
        
        # Construir prompt
        condition_info = ""
        if detected_conditions:
            strategies_info = []
            for cond in detected_conditions:
                strat = CONDITION_SEARCH_STRATEGIES.get(cond, {})
                desc = strat.get("description", cond)
                critical = ", ".join(strat.get("critical_vaccines", []))
                strategies_info.append(f"- {desc}: vacunas críticas: {critical}")
            condition_info = f"\n\nCONDICIONES DETECTADAS:\n" + "\n".join(strategies_info)
        
        prompt_template = ChatPromptTemplate.from_template("""
Eres un asistente especializado en vacunación para adultos en Colombia basado en guías oficiales (PAI Colombia, IDSA, ACIP/CDC).

{condition_info}

CONTEXTO DE GUÍAS OFICIALES:
{context}

PREGUNTA: {question}

INSTRUCCIONES CRÍTICAS:
1. Usa SOLO información del contexto proporcionado
2. CITA la fuente al final de cada recomendación: [FUENTE: nombre de vacuna - sección]
3. Si NO está en el contexto, di explícitamente "No tengo información suficiente sobre..."
4. NUNCA inventes dosis, esquemas, contraindicaciones o intervalos
5. Si detectaste condiciones específicas, PRIORIZA las vacunas críticas para esas condiciones
6. Para adultos ≥65 años, SIEMPRE menciona Neumococo PCV13+PPSV23 y Herpes Zóster si están en el contexto

Responde de manera clara, estructurada y profesional.
""")
        
        chain = prompt_template | self.llm
        
        try:
            response = chain.invoke({
                "context": context,
                "question": question,
                "condition_info": condition_info
            })
            
            answer_text = response.content
            
            # Determinar confianza
            confidence = "high" if len(relevant_docs) >= 5 else "medium" if len(relevant_docs) >= 2 else "low"
            
            # Extraer fuentes
            sources = []
            for doc in relevant_docs[:10]:
                sources.append(SourceInfo(
                    vaccine=doc.metadata.get('vaccine', 'Desconocida'),
                    section=doc.metadata.get('section', 'Desconocida'),
                    content_preview=doc.page_content[:200] + "...",
                    source_file=doc.metadata.get('source_file', 'Desconocido')
                ))
            
            return ChatResponse(
                answer=answer_text,
                confidence=confidence,
                sources=sources,
                recommendations=None,
                timestamp=datetime.now().isoformat()
            )
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error en generación: {str(e)}")

# Inicializar sistema
try:
    rag_system = VaccineRAGSystem()
    print("✅ Sistema RAG inicializado correctamente")
except Exception as e:
    print(f"❌ Error inicializando RAG: {e}")
    raise

# ============================================================================
# ENDPOINTS
# ============================================================================

@app.get("/")
async def root():
    """Endpoint raíz con información del servicio"""
    return {
        "service": "VaccinIA v3.3 - Communication Module API",
        "status": "active",
        "features": [
            "Búsqueda especializada por condición médica",
            "Embarazo, VIH, Cáncer, Trasplantes, Asplenia, Diabetes, EPOC, ERC, Adultos Mayores",
            "Anti-alucinación estricta",
            "Citación obligatoria de fuentes"
        ],
        "conditions_supported": list(CONDITION_SEARCH_STRATEGIES.keys())
    }

@app.post("/chat", response_model=ChatResponse)
async def chat(query: VaccinationQuery):
    """
    Endpoint principal para consultas de vacunación
    """
    try:
        return rag_system.answer_question(query.question, query.patient_profile)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health():
    """Health check"""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
