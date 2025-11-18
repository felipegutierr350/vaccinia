"""
VaccinIA v3.3 - Communication Module - API Backend con RAG
NUEVO: Sistema de búsqueda especializada por condición médica
- Embarazo, VIH, Cáncer, Trasplantes, Asplenia, Diabetes, EPOC, ERC, etc.
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any
import json
from datetime import datetime
import os

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
            "Herpes Zóster Shingrix 2 dosis"
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
            "Neumococo trasplante hematopoyético",
            "revacunación esquema completo trasplante médula"
        ],
        "timing_note": "Revacunación completa 6-12 meses post-trasplante",
        "description": "Receptores de trasplante de células madre hematopoyéticas"
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
            "Neumococo asplenia hiposplenia esplenectomía",
            "Meningococo ACYW asplenia capsulados",
            "Meningococo B Bexsero asplenia",
            "Haemophilus influenzae tipo b asplenia"
        ],
        "urgency": "ALTA - Riesgo sepsis fulminante por encapsulados",
        "timing_note": "Idealmente 2 semanas antes de esplenectomía electiva",
        "description": "Asplenia anatómica o funcional"
    },
    
    "diabetes": {
        "critical_vaccines": [
            "Neumococo PCV13", "Neumococo PPSV23",
            "Influenza", "COVID-19", "Hepatitis B"
        ],
        "k_docs": 12,
        "forced_queries": [
            "Neumococo diabetes mellitus tipo 1 tipo 2",
            "Influenza diabetes complicaciones",
            "Hepatitis B diabetes",
            "COVID-19 diabetes comorbilidad"
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
            "Neumococo EPOC enfermedad pulmonar obstructiva",
            "Influenza EPOC exacerbación",
            "COVID-19 EPOC comorbilidad respiratoria"
        ],
        "description": "Enfermedad pulmonar obstructiva crónica"
    },
    
    "erc": {
        "critical_vaccines": [
            "Hepatitis B", "Neumococo PCV13", "Neumococo PPSV23",
            "Influenza", "COVID-19"
        ],
        "k_docs": 12,
        "forced_queries": [
            "Hepatitis B enfermedad renal crónica hemodiálisis",
            "Neumococo insuficiencia renal",
            "vacunación diálisis ERC",
            "Hepatitis B dosis doble esquema renal"
        ],
        "timing_note": "Hepatitis B puede requerir esquema de dosis dobles",
        "description": "Enfermedad renal crónica / Hemodiálisis"
    },
    
    "hepatopatia": {
        "critical_vaccines": [
            "Hepatitis A", "Hepatitis B",
            "Neumococo PCV13", "Neumococo PPSV23",
            "Influenza", "COVID-19"
        ],
        "k_docs": 12,
        "forced_queries": [
            "Hepatitis A cirrosis hepatopatía crónica",
            "Hepatitis B enfermedad hepática",
            "Neumococo cirrosis"
        ],
        "description": "Enfermedad hepática crónica / Cirrosis"
    },
    
    "inmunosupresion": {
        "critical_vaccines": [
            "Neumococo PCV13", "Neumococo PPSV23",
            "Influenza", "COVID-19"
        ],
        "k_docs": 12,
        "contraindicated": ["vacunas vivas"],
        "forced_queries": [
            "inmunosupresión corticoides altas dosis",
            "biológicos anti-TNF rituximab vacunación",
            "contraindicación vacunas vivas inmunosupresores",
            "metotrexate azatioprina vacunación"
        ],
        "timing_note": "Vacunas vivas contraindicadas. Preferir vacunación antes de inicio de inmunosupresores",
        "description": "Inmunosupresión por medicamentos (no cáncer/VIH/trasplante)"
    },
    
    "enfermedad_autoinmune": {
        "critical_vaccines": [
            "Neumococo PCV13", "Neumococo PPSV23",
            "Influenza", "COVID-19"
        ],
        "k_docs": 12,
        "contraindicated": ["vacunas vivas si en tratamiento"],
        "forced_queries": [
            "enfermedad autoinmune lupus artritis reumatoide",
            "biológicos anti-TNF vacunación",
            "Neumococo enfermedad autoinmune"
        ],
        "description": "Enfermedades autoinmunes (LES, AR, etc.)"
    }
}

# Mapeo de keywords a condiciones
CONDITION_KEYWORDS = {
    "embarazo": ['embaraz', 'gestante', 'gestación', 'prenatal'],
    "adulto_mayor": ['adulto mayor', 'adulta mayor', 'tercera edad', '65 años', '66 años', '67 años', '68 años', '69 años', '70 años', '71 años', '72 años', '73 años', '74 años', '75 años', '76 años', '77 años', '78 años', '79 años', '80 años'],
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
# SISTEMA RAG
# ============================================================================

class VaccinIARAG:
    def __init__(self):
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        self.llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
        self.vectorstore = None
        self.knowledge_base = None
        
    def load_knowledge_base(self, json_path: str = "./vaccines_knowledge_base.json"):
        """Carga la base de conocimiento desde JSON"""
        print(f"📚 Cargando base de conocimiento desde {json_path}")
        with open(json_path, 'r', encoding='utf-8') as f:
            self.knowledge_base = json.load(f)
        print(f"✅ {len(self.knowledge_base['chunks'])} chunks cargados")
        
    def load_vectorstore(self, persist_dir: str = "./chroma_vaccinia"):
        """Carga el vector store existente - con detección de cambios en JSON"""
        import hashlib
        import shutil
        
        # Calcular hash del JSON actual
        kb_path = "./vaccines_knowledge_base.json"
        hash_file = os.path.join(persist_dir, "kb_hash.txt")
        
        current_hash = hashlib.md5(open(kb_path, 'rb').read()).hexdigest()
        
        # Verificar si necesita rebuild
        needs_rebuild = True
        if os.path.exists(persist_dir) and os.path.exists(hash_file):
            with open(hash_file, 'r') as f:
                stored_hash = f.read().strip()
            needs_rebuild = (current_hash != stored_hash)
            
            if needs_rebuild:
                print(f"🔄 JSON cambió (hash: {current_hash[:8]}... vs {stored_hash[:8]}...) - Eliminando ChromaDB antiguo")
                shutil.rmtree(persist_dir)
        
        if os.path.exists(persist_dir):
            print(f"📂 Cargando vector store desde {persist_dir}")
            self.vectorstore = Chroma(
                persist_directory=persist_dir,
                embedding_function=self.embeddings
            )
            print(f"✅ Vector store cargado")
        else:
            print(f"🔨 ChromaDB no existe - Creando desde cero...")
            os.makedirs(persist_dir, exist_ok=True)
            
            # Crear documentos desde knowledge base
            if self.knowledge_base is None:
                raise Exception("Knowledge base no cargada. Llama load_knowledge_base() primero.")
            
            documents = []
            for chunk in self.knowledge_base['chunks']:
                doc = Document(
                    page_content=chunk['content'],
                    metadata=chunk.get('metadata', {})
                )
                documents.append(doc)
            
            print(f"📚 Creando vector store con {len(documents)} documentos...")
            self.vectorstore = Chroma.from_documents(
                documents=documents,
                embedding=self.embeddings,
                persist_directory=persist_dir
            )
            
            # Guardar hash
            with open(hash_file, 'w') as f:
                f.write(current_hash)
            
            print(f"✅ Vector store creado y guardado en {persist_dir}")
    
    def _deduplicate(self, docs: List[Document]) -> List[Document]:
        """Elimina documentos duplicados basándose en contenido"""
        seen = set()
        unique_docs = []
        for doc in docs:
            content_hash = hash(doc.page_content[:200])
            if content_hash not in seen:
                seen.add(content_hash)
                unique_docs.append(doc)
        return unique_docs
    
    def detect_conditions(self, question: str, patient_profile: Optional[PatientProfile] = None) -> List[str]:
        """
        Detecta condiciones médicas en la pregunta o perfil del paciente
        """
        question_lower = question.lower()
        conditions = []
        
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
        
        # Eliminar duplicados manteniendo orden
        seen = set()
        unique_conditions = []
        for c in conditions:
            if c not in seen:
                seen.add(c)
                unique_conditions.append(c)
        
        return unique_conditions
    
    def retrieve_condition_docs(self, question: str, condition: str) -> List[Document]:
        """
        Búsqueda especializada por condición médica específica
        """
        if condition not in CONDITION_SEARCH_STRATEGIES:
            print(f"⚠️ Condición '{condition}' no tiene estrategia definida, usando búsqueda estándar")
            return self.vectorstore.similarity_search(question, k=8)
        
        strategy = CONDITION_SEARCH_STRATEGIES[condition]
        k = strategy.get('k_docs', 10)
        
        print(f"🔍 Usando estrategia '{condition}': {strategy.get('description', '')}")
        
        # 1. Búsqueda principal semántica
        docs_main = self.vectorstore.similarity_search(question, k=k//2)
        all_docs = list(docs_main)
        
        # 2. Búsquedas forzadas por queries específicas
        if 'forced_queries' in strategy:
            print(f"   🎯 Ejecutando {len(strategy['forced_queries'])} búsquedas forzadas")
            for forced_query in strategy['forced_queries']:
                try:
                    docs = self.vectorstore.similarity_search(forced_query, k=2)
                    all_docs.extend(docs)
                except Exception as e:
                    print(f"   ⚠️ Error en búsqueda forzada '{forced_query}': {e}")
        
        # 3. Forzado por metadata exacta
        if 'force_by_metadata' in strategy:
            print(f"   🎯 Forzando {len(strategy['force_by_metadata'])} vacunas por metadata")
            for vaccine_name in strategy['force_by_metadata']:
                try:
                    result = self.vectorstore.get(where={"vaccine": vaccine_name})
                    if result and 'documents' in result:
                        for i, doc_content in enumerate(result['documents']):
                            doc = Document(
                                page_content=doc_content,
                                metadata=result['metadatas'][i]
                            )
                            all_docs.append(doc)
                            print(f"      ✅ Forzado: {vaccine_name}")
                except Exception as e:
                    print(f"   ⚠️ Error forzando {vaccine_name}: {e}")
        
        # 4. Buscar contraindicaciones si aplica
        if 'contraindicated' in strategy:
            print(f"   ⚠️ Buscando contraindicaciones para {len(strategy['contraindicated'])} vacunas")
            for contraind in strategy['contraindicated'][:3]:
                try:
                    docs = self.vectorstore.similarity_search(
                        f"contraindicación {contraind} {condition}",
                        k=1
                    )
                    all_docs.extend(docs)
                except Exception as e:
                    print(f"   ⚠️ Error buscando contraindicación '{contraind}': {e}")
        
        # 5. Deduplicar
        unique_docs = self._deduplicate(all_docs)
        
        print(f"   📊 {len(unique_docs)} documentos únicos recuperados (de {len(all_docs)} totales)")
        
        return unique_docs[:k]
    
    def retrieve_pregnancy_docs(self, question: str) -> List[Document]:
        """Búsqueda especializada para embarazo (LEGACY - ahora usa retrieve_condition_docs)"""
        return self.retrieve_condition_docs(question, "embarazo")
    
    def retrieve_hiv_docs(self, question: str) -> List[Document]:
        """Búsqueda especializada para VIH (LEGACY - ahora usa retrieve_condition_docs)"""
        return self.retrieve_condition_docs(question, "vih")
    
    def build_prompt_template(self) -> ChatPromptTemplate:
        """Construye el template del prompt con instrucciones anti-alucinación"""
        
        system_template = """Eres VaccinIA, asistente médico especializado en vacunación para adultos en Colombia.

INSTRUCCIONES CRÍTICAS:
1. Responde ÚNICAMENTE basándote en las guías oficiales del PAI Colombia proporcionadas en el contexto
2. Si la información NO está en el contexto: "No tengo información suficiente en las guías del PAI Colombia para responder esto con precisión"
3. NUNCA inventes dosis, esquemas, contraindicaciones o intervalos
4. SIEMPRE cita la fuente: [FUENTE: Vacuna - Sección]
5. Recomienda consultar médico tratante si hay dudas o casos complejos
6. Sé preciso con números: dosis, intervalos, edades, recuentos CD4
7. Distingue claramente "FUERTE" vs "CONDICIONAL"
8. Si hay contraindicaciones, explícalas claramente con condiciones específicas

CORRECCIONES CONOCIDAS:
- VPH en VIH: 3 dosis (0, 1-2, 6 meses), NO más dosis
- Meningococo B: Bexsero, 2 dosis (0.5 ml IM), intervalo 1-2 meses, FUERTE en VIH

CONDICIONES ESPECIALES:
- Cáncer en quimioterapia: Vacunas vivas CONTRAINDICADAS, neumococo e influenza CRÍTICAS
- Asplenia: Riesgo ALTO sepsis por encapsulados (neumococo, meningococo, Hib)
- Trasplantes: Vacunas vivas CONTRAINDICADAS, revacunación completa en algunos casos
- Diabetes/EPOC/ERC: Neumococo e influenza especialmente importantes

Contexto PAI Colombia:
{context}

Perfil paciente:
{patient_context}
"""
        
        human_template = """Pregunta: {question}"""
        
        return ChatPromptTemplate.from_messages([
            ("system", system_template),
            ("human", human_template)
        ])
    
    def answer_question(
        self, 
        question: str, 
        patient_profile: Optional[PatientProfile] = None,
        k_docs: int = 8
    ) -> Dict[str, Any]:
        """
        Responde pregunta con detección automática de condiciones y búsqueda especializada
        """
        
        # DETECTAR CONDICIONES
        conditions = self.detect_conditions(question, patient_profile)
        
        print(f"\n{'='*60}")
        print(f"❓ Pregunta: {question[:100]}...")
        print(f"🔍 Condiciones detectadas: {conditions if conditions else 'Ninguna (búsqueda estándar)'}")
        print(f"{'='*60}\n")
        
        # SELECCIONAR ESTRATEGIA DE BÚSQUEDA
        if not conditions:
            # Búsqueda estándar
            docs = self.vectorstore.similarity_search(question, k=k_docs)
            search_type = "standard"
            print(f"📊 Búsqueda estándar: {len(docs)} documentos")
        
        elif 'embarazo' in conditions:
            docs = self.retrieve_condition_docs(question, 'embarazo')
            search_type = "embarazo"
        
        elif 'vih' in conditions:
            docs = self.retrieve_condition_docs(question, 'vih')
            search_type = "vih"
        
        else:
            # Usar estrategia de la primera condición detectada
            primary_condition = conditions[0]
            docs = self.retrieve_condition_docs(question, primary_condition)
            search_type = f"condition:{primary_condition}"
        
        # Construir contexto
        context = "\n\n---\n\n".join([doc.page_content for doc in docs])
        
        # Construir contexto del paciente
        patient_context = "No especificado"
        if patient_profile:
            patient_context = f"""
            Edad: {patient_profile.age}
            Sexo: {patient_profile.sex}
            Embarazo: {'Sí' if patient_profile.pregnant else 'No'}
            Inmunocomprometido: {'Sí' if patient_profile.immunocompromised else 'No'}
            Condiciones crónicas: {patient_profile.chronic_conditions or 'Ninguna'}
            """
        
        # Determinar vacunas críticas según condición
        critical_info = ""
        if conditions and conditions[0] in CONDITION_SEARCH_STRATEGIES:
            strategy = CONDITION_SEARCH_STRATEGIES[conditions[0]]
            if 'critical_vaccines' in strategy:
                critical_vaccines = ", ".join(strategy['critical_vaccines'])
                critical_info = f"\n\n🚨 VACUNAS CRÍTICAS para {conditions[0]}: {critical_vaccines}\nDEBES mencionar estas vacunas si están indicadas para el caso específico."
        
        # Crear chain y ejecutar
        prompt = self.build_prompt_template()
        chain = prompt | self.llm
        
        response = chain.invoke({
            "context": context + critical_info,
            "patient_context": patient_context,
            "question": question
        })
        
        # Determinar nivel de confianza
        confidence = "high" if len(docs) >= 5 else "medium" if len(docs) >= 3 else "low"
        
        # Preparar fuentes
        sources = []
        for doc in docs[:10]:  # Top 10 fuentes
            sources.append({
                "vaccine": doc.metadata.get('vaccine', 'Desconocida'),
                "section": doc.metadata.get('section', 'Desconocida'),
                "content_preview": doc.page_content[:300] + "...",
                "source_file": doc.metadata.get('source_file', 'Desconocido')
            })
        
        print(f"\n✅ Respuesta generada | Confidence: {confidence} | Fuentes: {len(sources)}")
        print(f"{'='*60}\n")
        
        return {
            "answer": response.content,
            "confidence": confidence,
            "sources": sources,
            "recommendations": None,
            "timestamp": datetime.now().isoformat()
        }

# ============================================================================
# INICIALIZACIÓN
# ============================================================================

app = FastAPI(
    title="VaccinIA v3.3 - Communication Module API",
    description="Sistema inteligente de recomendaciones de vacunación con búsqueda especializada por condición",
    version="3.2.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

rag_system = VaccinIARAG()

@app.on_event("startup")
async def startup_event():
    """Inicializa el sistema RAG al arrancar"""
    print("🚀 Iniciando VaccinIA v3.3 - Communication Module...")
    rag_system.load_knowledge_base()
    rag_system.load_vectorstore()
    print(f"✅ VaccinIA v3.3 - Communication Module listo con {len(rag_system.knowledge_base['chunks'])} chunks")

# ============================================================================
# ENDPOINTS
# ============================================================================

@app.get("/")
async def root():
    return {
        "service": "VaccinIA v3.3 - Communication Module API",
        "status": "active",
        "features": [
            "Búsqueda especializada por condición médica",
            "Embarazo, VIH, Cáncer, Trasplantes, Asplenia, Diabetes, EPOC, ERC",
            "Anti-alucinación estricta",
            "Citación obligatoria de fuentes"
        ],
        "conditions_supported": list(CONDITION_SEARCH_STRATEGIES.keys())
    }

@app.post("/chat", response_model=ChatResponse)
async def chat(query: VaccinationQuery):
    """
    Endpoint principal para consultas de vacunación
    Detecta automáticamente condiciones médicas y aplica búsqueda especializada
    """
    try:
        result = rag_system.answer_question(
            question=query.question,
            patient_profile=query.patient_profile,
            k_docs=8
        )
        return ChatResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/recommend")
async def recommend_vaccines(patient: PatientProfile):
    """
    Genera recomendaciones completas basadas en perfil del paciente
    """
    try:
        query = f"""Basándote en el siguiente perfil de paciente:
        Edad: {patient.age} años
        Sexo: {patient.sex}
        Embarazo: {'Sí' if patient.pregnant else 'No'}
        Inmunocomprometido: {'Sí' if patient.immunocompromised else 'No'}
        Condiciones crónicas: {patient.chronic_conditions or 'Ninguna'}
        
        ¿Qué vacunas están recomendadas según las guías del PAI Colombia?
        Incluye esquemas, dosis, intervalos y contraindicaciones si aplican.
        """
        
        result = rag_system.answer_question(
            question=query,
            patient_profile=patient,
            k_docs=8
        )
        
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/vaccines")
async def list_vaccines():
    """Lista todas las vacunas disponibles en la base de conocimiento"""
    vaccines = set()
    for chunk in rag_system.knowledge_base['chunks']:
        vaccine_name = chunk['metadata'].get('vaccine')
        if vaccine_name:
            vaccines.add(vaccine_name)
    
    return {
        "total": len(vaccines),
        "vaccines": sorted(list(vaccines))
    }

@app.get("/conditions")
async def list_conditions():
    """Lista todas las condiciones médicas con búsqueda especializada"""
    conditions_info = []
    
    for condition, strategy in CONDITION_SEARCH_STRATEGIES.items():
        conditions_info.append({
            "condition": condition,
            "description": strategy.get('description', ''),
            "critical_vaccines": strategy.get('critical_vaccines', []),
            "urgency": strategy.get('urgency', 'NORMAL'),
            "timing_note": strategy.get('timing_note', None)
        })
    
    return {
        "total": len(conditions_info),
        "conditions": conditions_info
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "version": "3.2.0",
        "vectorstore": "loaded" if rag_system.vectorstore else "not_loaded",
        "knowledge_base": "loaded" if rag_system.knowledge_base else "not_loaded"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

# Force rebuild lunes, 17 de noviembre de 2025, 22:01:11 -05
# Force rebuild lunes, 17 de noviembre de 2025, 22:05:56 -05
