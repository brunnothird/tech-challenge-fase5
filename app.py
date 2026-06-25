import streamlit as st
import pandas as pd
import numpy as np
import pickle

# 1. Configuração Inicial da Página
st.set_page_config(
    page_title="Passos Mágicos - Sistema Preditivo",
    page_icon="🪄",
    layout="centered"
)

# 2. Carregar o Modelo Preditivo Treinado
@st.cache_resource
def carregar_modelo():
    with open("modelo_risco_passos.pkl", "rb") as f:
        return pickle.load(f)

try:
    modelo = carregar_modelo()
except Exception as e:
    st.error("Erro ao carregar o modelo preditivo. Certifique-se de que o arquivo 'modelo_risco_passos.pkl' está no mesmo diretório.")

# 3. Interface Visual (Cabeçalho)
st.title("🪄 Portal de Prevenção de Defasagem - Passos Mágicos")
st.markdown("""
Esta aplicação utiliza um modelo de **Inteligência Artificial (Machine Learning)** para prever o risco 
de defasagem escolar e guiar as equipes de assistência psicopedagógica nas tomadas de decisão preventivas.
""")

st.write("---")
st.subheader("📊 Insira os Indicadores Multidimensionais do Aluno:")

# 4. Criando os Sliders para Entrada de Dados (Notas de 0 a 10)
# Os nomes das variáveis e os limites devem bater exatamente com o que o modelo aprendeu
ian = st.slider("📐 IAN - Adequação do Nível (Idade x Série)", 0.0, 10.0, 7.0, step=0.1)
ida = st.slider("📝 IDA - Desempenho Acadêmico (Notas)", 0.0, 10.0, 7.0, step=0.1)
ieg = st.slider("🔥 IEG - Engajamento nas Atividades", 0.0, 10.0, 7.0, step=0.1)
iaa = st.slider("👤 IAA - Autoavaliação do Aluno", 0.0, 10.0, 7.0, step=0.1)
ips = st.slider("🧠 IPS - Aspectos Psicossociais", 0.0, 10.0, 7.0, step=0.1)
ipp = st.slider("🩺 IPP - Aspectos Psicopedagógicos", 0.0, 10.0, 7.0, step=0.1)

st.write("---")

# 5. Processamento da Predição ao Clicar no Botão
if st.button("🔮 Avaliar Risco do Aluno"):
    # Organiza os dados informados no formato que o modelo espera (DataFrame)
    dados_aluno = pd.DataFrame([[ian, ida, ieg, iaa, ips, ipp]], 
                               columns=['IAN', 'IDA', 'IEG', 'IAA', 'IPS', 'IPP'])
    
    # Realiza a predição (0 ou 1) e calcula a probabilidade percentual
    predicao = modelo.predict(dados_aluno)[0]
    probabilidade = modelo.predict_proba(dados_aluno)[0][1] * 100
    
    # 6. Exibição Customizada dos Resultados para a Equipe da ONG
    if predicao == 1:
        st.error(f"🚨 **ALERTA DE RISCO DETECTADO!**")
        st.metric(label="Probabilidade de Defasagem/Evasão", value=f"{probabilidade:.1f}%")
        st.warning("👉 **Recomendação:** Direcionar o aluno imediatamente para uma avaliação prioritária com o time psicopedagógico e reforço escolar focado.")
    else:
        st.success(f"✅ **ALUNO EM SITUAÇÃO ESTÁVEL**")
        st.metric(label="Probabilidade de Risco", value=f"{probabilidade:.1f}%")
        st.info("🟢 O aluno apresenta ótimos índices de engajamento e desempenho acadêmico. Continue acompanhando os ciclos regulares.")
