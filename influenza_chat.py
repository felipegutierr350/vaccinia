"""
VACCINIA INFLUENZA - SISTEMA HÍBRIDO PROFESIONAL
Módulo de chat conversacional para vacunación contra Influenza
Basado en literatura Q1/Q2 (Lancet, JAMA, Cochrane, etc.)
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
from enum import Enum
import openai
import os
import json
from datetime import datetime
import hashlib

router = APIRouter(prefix="/influenza", tags=["Influenza"])

# ============================================================
# CONFIGURACIÓN
# ============================================================

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if OPENAI_API_KEY:
    openai.api_key = OPENAI_API_KEY

# ============================================================
# MODELOS PYDANTIC
# ============================================================

class ConversationState(str, Enum):
    WELCOME = "welcome"
    ASK_AGE = "ask_age"
    ASK_CONDITIONS = "ask_conditions"
    ASK_EXPOSURE = "ask_exposure"
    ASK_PREVIOUS_VAX = "ask_previous_vax"
    RECOMMENDATION = "recommendation"
    FREE_CHAT = "free_chat"

class UserProfile(BaseModel):
    age: Optional[int] = None
    conditions: List[str] = []
    high_exposure: Optional[bool] = None
    previous_vaccination: Optional[str] = None

class ActiveProfiles(BaseModel):
    adult_mayor: bool = False
    chronic_disease: bool = False
    high_circulation: bool = False

class Session(BaseModel):
    session_id: str
    state: ConversationState = ConversationState.WELCOME
    user_profile: UserProfile = UserProfile()
    active_profiles: ActiveProfiles = ActiveProfiles()
    chat_history: List[Dict[str, str]] = []
    created_at: str = ""

class ChatRequest(BaseModel):
    session_id: Optional[str] = None
    message: str

class ChatResponse(BaseModel):
    session_id: str
    response: str
    state: str
    active_profiles: Dict[str, bool]
    options: Optional[List[str]] = None

# ============================================================
# ALMACENAMIENTO DE SESIONES
# ============================================================

sessions: Dict[str, Session] = {}

def get_or_create_session(session_id: Optional[str] = None) -> Session:
    if session_id and session_id in sessions:
        return sessions[session_id]
    new_id = hashlib.md5(f"{datetime.now().isoformat()}".encode()).hexdigest()[:12]
    session = Session(session_id=new_id, created_at=datetime.now().isoformat())
    sessions[new_id] = session
    return session

# ============================================================
# SCRIPTS BASE (Material basal - NO SE MODIFICAN)
# ============================================================

SCRIPTS = {
    "A1_intro": """A partir de los 65 años, una gripe puede ser mucho más que un malestar: aumenta el riesgo de neumonía, hospitalización y pérdida de independencia.

La vacuna contra la influenza reduce claramente estas complicaciones y es una de las medidas más efectivas para proteger tu salud cada temporada.""",

    "A2_reservada": """Revisando tu información, vemos que hay una dosis de la vacuna contra la influenza que podría estar disponible para ti esta semana.

Este refuerzo ayuda a que tu sistema inmune responda mejor frente a los virus de esta temporada.

¿Te gustaría saber más sobre dónde y cuándo aplicártela?""",

    "A3_familia": """Sabemos que disfrutar de la familia es importante para ti.

Vacunarte no solo reduce el riesgo de neumonía y hospitalización, también crea un escudo que protege a tus nietos y familiares más vulnerables.

Es una forma sencilla y segura de compartir tiempo juntos sin preocupaciones.""",

    "A4_sano": """Aunque te sientas bien, con la edad el cuerpo responde de forma diferente a la gripe.

Incluso personas sanas mayores de 65 años tienen más riesgo de hospitalización y complicaciones.

La vacuna reduce claramente ese riesgo y suele causar solo molestias leves como dolor en el brazo o algo de cansancio uno o dos días.""",

    "B1_intro": """En personas con enfermedades crónicas, la gripe no es solo un resfriado.

Puede descompensar la enfermedad de base y provocar crisis graves o ingresos hospitalarios.

Por eso, la vacuna contra la influenza es parte del cuidado habitual de tu salud.""",

    "B2_corazon": """Si tienes problemas del corazón, una gripe puede desencadenar un infarto o una descompensación cardíaca.

La vacuna contra la influenza reduce el riesgo de estos eventos y forma parte del cuidado del corazón, igual que tus medicamentos diarios.""",

    "B3_diabetes": """Con diabetes, una gripe puede descontrolar el azúcar y llevar a una hospitalización.

Vacunarte reduce ese riesgo y está recomendada específicamente para personas con diabetes como parte del cuidado integral.""",

    "B4_epoc_asma": """En personas con EPOC o asma, la gripe puede provocar crisis respiratorias importantes, necesidad de oxígeno o ingreso hospitalario.

La vacuna anual disminuye claramente ese riesgo y ayuda a mantener la respiración estable.""",

    "B5_medicamentos": """Es una duda muy válida.

La vacuna contra la gripe es segura y no interfiere con tus medicamentos.

Al contrario, evita el estrés que una infección grave puede causar en tu cuerpo y ayuda a que tu tratamiento funcione mejor.""",

    "C1_intro": """En tu entorno hay mucha interacción diaria entre personas, lo que facilita que la gripe se transmita con mayor rapidez.

Vacunarte reduce tu riesgo de enfermar y también ayuda a proteger a quienes te rodean.""",

    "C2_norma_social": """Muchas personas en entornos similares al tuyo ya se están vacunando para reducir el riesgo de contagio.

Al unirte a ellos, te proteges y contribuyes a que el virus circule menos.""",

    "C3_cortafuegos": """Cuando te vacunas, actúas como un cortafuegos: evitas que el virus llegue a ti y que pase de ti a tu familia o compañeros.

Es una acción rápida y que protege a toda la comunidad.""",

    "T1_seguridad": """Los efectos secundarios más comunes son dolor en el brazo, algo de cansancio o febrícula durante uno o dos días.

Las reacciones graves son muy raras.

La vacuna no puede causar gripe porque es una vacuna inactivada.""",

    "T2_anual": """El virus de la influenza cambia cada temporada y la protección disminuye con el tiempo.

Por eso la vacuna se recomienda una vez al año.""",

    "T3_cierre_profesional": """Como sistema de salud, la recomendación es clara: vacunarte reduce el riesgo de hospitalización y complicaciones graves.

Es una medida sencilla, segura y respaldada por la mejor evidencia disponible.

¿Hay algo más en lo que pueda ayudarte?"""
}

FLOW_MESSAGES = {
    "welcome": """¡Hola! 👋 Soy Vaccinia, tu asistente de vacunación contra la influenza.

Para darte la mejor recomendación personalizada, necesito hacerte unas preguntas rápidas.

¿Empezamos?""",

    "ask_age": """**Pregunta 1 de 4**

¿Cuántos años tienes?""",

    "ask_conditions": """**Pregunta 2 de 4**

¿Tienes alguna de estas condiciones de salud?

- Problemas del corazón
- Diabetes
- Asma o EPOC
- Otra enfermedad crónica
- Ninguna""",

    "ask_exposure": """**Pregunta 3 de 4**

¿Vives o trabajas en un lugar con mucho contacto con personas?

Por ejemplo: transporte público, comercio, atención al público, hospitales, escuelas, oficinas concurridas.""",

    "ask_previous_vax": """**Pregunta 4 de 4**

¿Te vacunaste contra la gripe el año pasado?"""
}

# ============================================================
# MOTOR DE DECISIÓN
# ============================================================

def activate_profiles(profile: UserProfile) -> ActiveProfiles:
    active = ActiveProfiles()
    if profile.age and profile.age >= 65:
        active.adult_mayor = True
    chronic_conditions = ["corazon", "diabetes", "epoc", "asma", "cronica", "otra"]
    if any(cond.lower() in " ".join(profile.conditions).lower() for cond in chronic_conditions):
        active.chronic_disease = True
    if profile.high_exposure:
        active.high_circulation = True
    return active

def select_scripts(active: ActiveProfiles, profile: UserProfile) -> List[str]:
    scripts_to_use = []
    if active.adult_mayor:
        scripts_to_use.append("A1_intro")
        if profile.previous_vaccination == "no":
            scripts_to_use.append("A2_reservada")
        scripts_to_use.append("A3_familia")
    if active.chronic_disease:
        scripts_to_use.append("B1_intro")
        conditions_lower = " ".join(profile.conditions).lower()
        if "corazon" in conditions_lower or "cardia" in conditions_lower:
            scripts_to_use.append("B2_corazon")
        if "diabetes" in conditions_lower:
            scripts_to_use.append("B3_diabetes")
        if "epoc" in conditions_lower or "asma" in conditions_lower:
            scripts_to_use.append("B4_epoc_asma")
    if active.high_circulation:
        scripts_to_use.append("C1_intro")
        scripts_to_use.append("C3_cortafuegos")
    scripts_to_use.append("T3_cierre_profesional")
    return scripts_to_use

# ============================================================
# GENERADOR DE RESPUESTAS CON LLM
# ============================================================

def generate_personalized_response(session: Session, scripts_to_use: List[str], user_message: str = "") -> str:
    profile = session.user_profile
    active = session.active_profiles
    
    context_parts = []
    if profile.age:
        context_parts.append(f"Edad: {profile.age} años")
    if profile.conditions:
        context_parts.append(f"Condiciones: {', '.join(profile.conditions)}")
    if profile.high_exposure is not None:
        context_parts.append(f"Alta exposición: {'Sí' if profile.high_exposure else 'No'}")
    if profile.previous_vaccination:
        vax_text = {"si": "Sí", "no": "No", "no_recuerdo": "No recuerda"}
        context_parts.append(f"Vacunación previa: {vax_text.get(profile.previous_vaccination, profile.previous_vaccination)}")
    
    user_context = "\n".join(context_parts)
    
    active_text = []
    if active.adult_mayor:
        active_text.append("ADULTO MAYOR (≥65 años) - Prioridad alta")
    if active.chronic_disease:
        active_text.append("ENFERMEDAD CRÓNICA - Prioridad alta")
    if active.high_circulation:
        active_text.append("ALTA EXPOSICIÓN - Prioridad moderada")
    
    scripts_content = "\n\n---\n\n".join([f"**{key}:**\n{SCRIPTS[key]}" for key in scripts_to_use if key in SCRIPTS])
    
    system_prompt = f"""Eres Vaccinia, un asistente experto en vacunación contra influenza.
Tu rol es ayudar a las personas a entender por qué vacunarse puede proteger su salud.

REGLAS ESTRICTAS:
1. USA los scripts base proporcionados como fundamento de tu respuesta
2. PERSONALIZA el tono para ser cálido y empático
3. NO INVENTES datos médicos, estadísticas ni cifras
4. NO uses lenguaje técnico innecesario
5. Mantén respuestas concisas (máximo 3-4 párrafos)
6. Si el usuario hace preguntas, responde manteniendo el contexto de su perfil

PERFIL DEL USUARIO:
{user_context}

PERFILES ACTIVOS:
{chr(10).join(active_text) if active_text else "Ninguno específico - Usuario general"}

SCRIPTS BASE A UTILIZAR (combina y personaliza):
{scripts_content}

HISTORIAL DE CONVERSACIÓN:
{json.dumps(session.chat_history[-6:], ensure_ascii=False) if session.chat_history else "Inicio de conversación"}
"""

    try:
        response = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message if user_message else "Dame la recomendación personalizada basada en mi perfil."}
            ],
            temperature=0.7,
            max_tokens=800
        )
        return response.choices[0].message.content
    except Exception as e:
        return "\n\n".join([SCRIPTS[key] for key in scripts_to_use[:2] if key in SCRIPTS])

def generate_free_chat_response(session: Session, user_message: str) -> str:
    profile = session.user_profile
    active = session.active_profiles
    
    context_parts = []
    if profile.age:
        context_parts.append(f"Edad: {profile.age} años")
    if profile.conditions:
        context_parts.append(f"Condiciones: {', '.join(profile.conditions)}")
    if active.adult_mayor:
        context_parts.append("Es adulto mayor (≥65)")
    if active.chronic_disease:
        context_parts.append("Tiene enfermedad crónica")
    
    system_prompt = f"""Eres Vaccinia, un asistente experto en vacunación contra influenza.
Estás en una conversación continua con un usuario que ya recibió su recomendación personalizada.

CONTEXTO DEL USUARIO:
{chr(10).join(context_parts)}

REGLAS:
1. Responde de forma cálida y empática
2. Mantén el contexto del perfil del usuario en tus respuestas
3. NO INVENTES datos médicos ni estadísticas
4. Si no sabes algo, dilo honestamente
5. Puedes usar estos datos de respaldo:
   - La vacuna es anual porque el virus muta
   - Efectos secundarios comunes: dolor en brazo, cansancio leve 1-2 días
   - La vacuna NO puede causar gripe (es inactivada)
   - Protección empieza ~2 semanas después de aplicar
   - Es segura con otros medicamentos

HISTORIAL RECIENTE:
{json.dumps(session.chat_history[-8:], ensure_ascii=False)}

Responde de forma concisa y útil."""

    try:
        response = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message}
            ],
            temperature=0.7,
            max_tokens=500
        )
        return response.choices[0].message.content
    except Exception as e:
        return "Disculpa, tuve un problema procesando tu pregunta. ¿Podrías reformularla?"

# ============================================================
# PROCESADORES DE RESPUESTAS
# ============================================================

def parse_age(message: str) -> Optional[int]:
    import re
    numbers = re.findall(r'\d+', message)
    if numbers:
        age = int(numbers[0])
        if 0 < age < 120:
            return age
    return None

def parse_conditions(message: str) -> List[str]:
    conditions = []
    message_lower = message.lower()
    if any(word in message_lower for word in ["corazón", "corazon", "cardiaco", "cardíaco", "cardia"]):
        conditions.append("corazón")
    if "diabetes" in message_lower:
        conditions.append("diabetes")
    if any(word in message_lower for word in ["epoc", "pulmonar", "pulmón", "pulmon"]):
        conditions.append("EPOC")
    if "asma" in message_lower:
        conditions.append("asma")
    if any(word in message_lower for word in ["otra", "otro", "crónica", "cronica"]):
        conditions.append("otra condición crónica")
    if any(word in message_lower for word in ["ninguna", "ninguno", "no tengo", "no", "nada"]):
        conditions = []
    return conditions

def parse_yes_no(message: str) -> Optional[bool]:
    message_lower = message.lower().strip()
    if any(word in message_lower for word in ["sí", "si", "yes", "claro", "correcto", "afirmativo", "mucho", "bastante"]):
        return True
    if any(word in message_lower for word in ["no", "nop", "nope", "negativo", "poco", "tranquilo"]):
        return False
    return None

def parse_previous_vax(message: str) -> Optional[str]:
    message_lower = message.lower().strip()
    if any(word in message_lower for word in ["sí", "si", "yes"]):
        return "si"
    if any(word in message_lower for word in ["no recuerdo", "no sé", "no se", "no estoy seguro"]):
        return "no_recuerdo"
    if "no" in message_lower:
        return "no"
    return None

# ============================================================
# ENDPOINT PRINCIPAL
# ============================================================

@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    session = get_or_create_session(request.session_id)
    user_message = request.message.strip()
    
    if user_message:
        session.chat_history.append({"role": "user", "content": user_message})
    
    response_text = ""
    options = None
    
    if session.state == ConversationState.WELCOME:
        response_text = FLOW_MESSAGES["welcome"]
        session.state = ConversationState.ASK_AGE
        options = ["¡Sí, empecemos!", "Tengo una duda primero"]
    
    elif session.state == ConversationState.ASK_AGE:
        if "duda" in user_message.lower():
            response_text = "Claro, ¿cuál es tu duda? Puedo responderla y luego continuamos."
        else:
            response_text = FLOW_MESSAGES["ask_age"]
            session.state = ConversationState.ASK_CONDITIONS
    
    elif session.state == ConversationState.ASK_CONDITIONS:
        age = parse_age(user_message)
        if age:
            session.user_profile.age = age
            response_text = FLOW_MESSAGES["ask_conditions"]
            session.state = ConversationState.ASK_EXPOSURE
            options = ["Problemas del corazón", "Diabetes", "Asma o EPOC", "Otra enfermedad crónica", "Ninguna"]
        else:
            response_text = "No pude entender tu edad. ¿Podrías escribirla en números? Por ejemplo: 45"
    
    elif session.state == ConversationState.ASK_EXPOSURE:
        conditions = parse_conditions(user_message)
        session.user_profile.conditions = conditions
        response_text = FLOW_MESSAGES["ask_exposure"]
        session.state = ConversationState.ASK_PREVIOUS_VAX
        options = ["Sí, tengo mucho contacto", "No, trabajo/vivo tranquilo"]
    
    elif session.state == ConversationState.ASK_PREVIOUS_VAX:
        high_exp = parse_yes_no(user_message)
        session.user_profile.high_exposure = high_exp if high_exp is not None else False
        response_text = FLOW_MESSAGES["ask_previous_vax"]
        session.state = ConversationState.RECOMMENDATION
        options = ["Sí, me vacuné", "No, no me vacuné", "No recuerdo"]
    
    elif session.state == ConversationState.RECOMMENDATION:
        prev_vax = parse_previous_vax(user_message)
        session.user_profile.previous_vaccination = prev_vax if prev_vax else "no_recuerdo"
        session.active_profiles = activate_profiles(session.user_profile)
        scripts_to_use = select_scripts(session.active_profiles, session.user_profile)
        response_text = generate_personalized_response(session, scripts_to_use)
        session.state = ConversationState.FREE_CHAT
    
    elif session.state == ConversationState.FREE_CHAT:
        response_text = generate_free_chat_response(session, user_message)
    
    session.chat_history.append({"role": "assistant", "content": response_text})
    sessions[session.session_id] = session
    
    return ChatResponse(
        session_id=session.session_id,
        response=response_text,
        state=session.state.value,
        active_profiles={
            "adulto_mayor": session.active_profiles.adult_mayor,
            "enfermedad_cronica": session.active_profiles.chronic_disease,
            "alta_exposicion": session.active_profiles.high_circulation
        },
        options=options
    )

@router.get("/session/{session_id}")
async def get_session(session_id: str):
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Sesión no encontrada")
    session = sessions[session_id]
    return {
        "session_id": session.session_id,
        "state": session.state.value,
        "user_profile": session.user_profile.dict(),
        "active_profiles": session.active_profiles.dict(),
        "messages_count": len(session.chat_history),
        "created_at": session.created_at
    }

@router.delete("/session/{session_id}")
async def delete_session(session_id: str):
    if session_id in sessions:
        del sessions[session_id]
        return {"message": "Sesión eliminada"}
    raise HTTPException(status_code=404, detail="Sesión no encontrada")

@router.get("/health")
async def health():
    return {
        "status": "healthy",
        "module": "influenza",
        "active_sessions": len(sessions),
        "scripts_loaded": len(SCRIPTS)
    }
