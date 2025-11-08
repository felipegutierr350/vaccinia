"""
VaccinIA v3.2 - Frontend Profesional
Sistema Inteligente de Recomendaciones de Vacunación
"""
import streamlit as st
import requests
import json
from datetime import datetime

# Configuración de página
st.set_page_config(
    page_title="VaccinIA - Recomendaciones de Vacunación",
    page_icon="💉",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personalizado
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .subtitle {
        text-align: center;
        color: #666;
        font-size: 1.2rem;
        margin-bottom: 2rem;
    }
    .success-box {
        padding: 1rem;
        background-color: #d4edda;
        border-left: 5px solid #28a745;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .info-box {
        padding: 1rem;
        background-color: #d1ecf1;
        border-left: 5px solid #17a2b8;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .warning-box {
        padding: 1rem;
        background-color: #fff3cd;
        border-left: 5px solid #ffc107;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .source-card {
        padding: 0.5rem;
        background-color: #f8f9fa;
        border-radius: 5px;
        margin: 0.5rem 0;
        font-size: 0.9rem;
    }
    .stButton>button {
        width: 100%;
        background-color: #1f77b4;
        color: white;
        font-weight: bold;
        padding: 0.5rem 1rem;
        border-radius: 5px;
    }
    .condition-badge {
        display: inline-block;
        padding: 0.3rem 0.6rem;
        background-color: #17a2b8;
        color: white;
        border-radius: 15px;
        margin: 0.2rem;
        font-size: 0.85rem;
    }
</style>
""", unsafe_allow_html=True)

# API URL (cambiar cuando esté en Railway)
API_URL = st.secrets.get("API_URL", "http://localhost:8000")

# Header
st.markdown('<div class="main-header">💉 VaccinIA v3.2</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Sistema Inteligente de Recomendaciones de Vacunación para Adultos en Colombia</div>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.image("https://via.placeholder.com/150x150.png?text=VaccinIA", width=150)
    st.markdown("---")
    
    st.markdown("### 📊 Estadísticas del Sistema")
    try:
        response = requests.get(f"{API_URL}/vaccines", timeout=5)
        if response.status_code == 200:
            data = response.json()
            st.metric("Vacunas disponibles", data.get("total", "N/A"))
    except:
        st.warning("Backend no disponible")
    
    st.markdown("### ℹ️ Información")
    st.info("""
    **VaccinIA** es un sistema basado en:
    - 📚 PAI Colombia
    - 🏥 IDSA 2025
    - 🔬 ACIP/CDC
    - ✅ ACR 2022
    
    **1,162 documentos** de guías oficiales
    """)
    
    st.markdown("---")
    st.markdown("### 🏥 Condiciones Soportadas")
    conditions = [
        "Embarazo", "VIH/SIDA", "Cáncer", 
        "Trasplantes", "Diabetes", "EPOC",
        "Enfermedad Renal", "Asplenia",
        "Enf. Autoinmunes", "Inmunosupresión"
    ]
    for cond in conditions:
        st.markdown(f"✅ {cond}")

# Tabs principales
tab1, tab2, tab3 = st.tabs(["💬 Consulta Rápida", "👤 Perfil de Paciente", "📖 Guías"])

# TAB 1: Consulta Rápida
with tab1:
    st.markdown("## 💬 Consulta Rápida")
    st.markdown("Pregunta directamente sobre vacunas, condiciones médicas o esquemas específicos.")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        query = st.text_area(
            "¿Qué necesitas saber sobre vacunación?",
            placeholder="Ejemplo: Paciente con trasplante renal de 6 meses, ¿qué vacunas respiratorias necesita?",
            height=100,
            key="query_input"
        )
    
    with col2:
        st.markdown("### Ejemplos rápidos")
        if st.button("🤰 Embarazo"):
            query = "¿Qué vacunas necesita una embarazada de 30 semanas?"
        if st.button("🦠 VIH"):
            query = "Paciente VIH CD4=150, ¿qué vacunas necesita?"
        if st.button("🎗️ Cáncer"):
            query = "¿Qué vacunas puede recibir paciente en quimioterapia?"
        if st.button("🫀 Trasplante"):
            query = "Candidato a trasplante renal, ¿cuándo vacunar Herpes Zóster?"
    
    if st.button("🔍 Consultar", type="primary"):
        if query:
            with st.spinner("🧠 Consultando guías oficiales..."):
                try:
                    response = requests.post(
                        f"{API_URL}/chat",
                        json={"question": query},
                        timeout=30
                    )
                    
                    if response.status_code == 200:
                        data = response.json()
                        
                        # Respuesta principal
                        st.markdown("### 📋 Recomendación")
                        st.markdown(f'<div class="success-box">{data["answer"]}</div>', unsafe_allow_html=True)
                        
                        # Nivel de confianza
                        confidence_color = {
                            "high": "🟢",
                            "medium": "🟡",
                            "low": "��"
                        }
                        st.markdown(f"**Confianza:** {confidence_color.get(data['confidence'], '⚪')} {data['confidence'].upper()}")
                        
                        # Fuentes
                        with st.expander(f"📚 Ver fuentes consultadas ({len(data['sources'])} documentos)"):
                            for i, source in enumerate(data['sources'][:5], 1):
                                st.markdown(f"""
                                <div class="source-card">
                                <strong>Fuente {i}:</strong> {source['vaccine']} - {source['section']}<br>
                                <small>{source['content_preview'][:200]}...</small>
                                </div>
                                """, unsafe_allow_html=True)
                        
                        # Timestamp
                        st.caption(f"Consultado: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                    else:
                        st.error(f"Error en la consulta: {response.status_code}")
                
                except Exception as e:
                    st.error(f"Error de conexión: {str(e)}")
                    st.info("Verifica que el backend esté corriendo")
        else:
            st.warning("Por favor escribe una pregunta")

# TAB 2: Perfil de Paciente
with tab2:
    st.markdown("## 👤 Recomendaciones por Perfil Completo")
    
    col1, col2 = st.columns(2)
    
    with col1:
        age = st.number_input("Edad", min_value=18, max_value=120, value=50)
        sex = st.selectbox("Sexo", ["M", "F"])
        pregnant = st.checkbox("Embarazo actual")
        immunocompromised = st.checkbox("Inmunocomprometido")
    
    with col2:
        st.markdown("### Condiciones crónicas")
        conditions_selected = st.multiselect(
            "Selecciona condiciones",
            [
                "Diabetes", "Hipertensión", "EPOC", "Asma",
                "Enfermedad renal crónica", "Cirrosis hepática",
                "Cáncer activo", "VIH/SIDA", "Trasplante previo",
                "Enfermedad autoinmune", "Asplenia"
            ]
        )
        
        other_conditions = st.text_input("Otras condiciones", placeholder="Ej: Tratamiento con rituximab")
    
    if st.button("📊 Generar Recomendaciones Completas", type="primary"):
        chronic_conditions = ", ".join(conditions_selected)
        if other_conditions:
            chronic_conditions += f", {other_conditions}"
        
        with st.spinner("🧠 Analizando perfil completo..."):
            try:
                profile_data = {
                    "age": age,
                    "sex": sex,
                    "pregnant": pregnant,
                    "immunocompromised": immunocompromised,
                    "chronic_conditions": chronic_conditions
                }
                
                response = requests.post(
                    f"{API_URL}/recommend",
                    json=profile_data,
                    timeout=30
                )
                
                if response.status_code == 200:
                    data = response.json()
                    
                    st.success("✅ Análisis completado")
                    
                    # Resumen del perfil
                    st.markdown("### 📋 Perfil del Paciente")
                    st.markdown(f"""
                    - **Edad:** {age} años
                    - **Sexo:** {sex}
                    - **Embarazo:** {'Sí' if pregnant else 'No'}
                    - **Inmunocomprometido:** {'Sí' if immunocompromised else 'No'}
                    - **Condiciones:** {chronic_conditions if chronic_conditions else 'Ninguna'}
                    """)
                    
                    # Recomendaciones
                    st.markdown("### 💉 Recomendaciones de Vacunación")
                    st.markdown(f'<div class="info-box">{data["answer"]}</div>', unsafe_allow_html=True)
                    
                    # Fuentes
                    with st.expander(f"📚 Fuentes consultadas ({len(data['sources'])})"):
                        for source in data['sources']:
                            st.markdown(f"- **{source['vaccine']}** ({source['section']})")
                
                else:
                    st.error("Error al generar recomendaciones")
            
            except Exception as e:
                st.error(f"Error: {str(e)}")

# TAB 3: Guías
with tab3:
    st.markdown("## 📖 Información sobre Vacunas")
    
    try:
        response = requests.get(f"{API_URL}/vaccines", timeout=5)
        if response.status_code == 200:
            vaccines_data = response.json()
            
            st.markdown(f"### {vaccines_data['total']} Vacunas Disponibles")
            
            cols = st.columns(3)
            for i, vaccine in enumerate(sorted(vaccines_data['vaccines'])):
                with cols[i % 3]:
                    st.markdown(f"""
                    <div class="source-card">
                    ✅ {vaccine}
                    </div>
                    """, unsafe_allow_html=True)
        
        # Información adicional
        st.markdown("---")
        st.markdown("### 🏥 Guías Integradas")
        st.markdown("""
        - 📚 **PAI Colombia** - Programa Ampliado de Inmunizaciones
        - 🏥 **IDSA 2025** - Infectious Diseases Society of America (Nov 2025)
        - 🔬 **ACIP/CDC** - Advisory Committee on Immunization Practices
        - ✅ **ACR 2022** - American College of Rheumatology
        - 🌍 **OMS/WHO** - Organización Mundial de la Salud
        """)
        
    except:
        st.error("No se pudo cargar la información de vacunas")

# Footer
st.markdown("---")
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("### ⚕️ Disclaimer")
    st.caption("Esta es una herramienta de apoyo clínico. Consulte siempre con un profesional de salud.")

with col2:
    st.markdown("### 📊 Versión")
    st.caption("VaccinIA v3.2 - Noviembre 2025")

with col3:
    st.markdown("### 👨‍⚕️ Desarrollado por")
    st.caption("Dr. Ivan Felipe Gutierrez - INP Colombia")
