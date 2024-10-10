import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression

# Carregando os dados
df = pd.read_csv("dados_saude.csv")

# Instanciando o modelo
modelo = LinearRegression()

# Definindo variáveis independentes (x) e dependentes (y)
x = df[["idade", "atividade"]]
y = df["colesterol"]

# Treinando o modelo
modelo.fit(x, y)

# Configurando o estilo da página
st.markdown(
    """
    <style>
    .main {
        background-image: url('https://images.unsplash.com/photo-1518611012118-696072aa579a?q=80&w=2070&auto=format&fit=crop&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D');
        background-color: rgba(255, 255, 255, 0.5);  /* Cor de fundo (opcional) */
        background-blend-mode: lighten;
        background-size: cover;
    }
    .titulo {
        color: #000000;  /* Cor preta para o título */
        text-align: center;  /* Centraliza o texto */
    }
    .input-label {
        color: #000000;  /* Cor preta para o texto dos labels */
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Título da aplicação
st.markdown("<h1 class='titulo'>Consulte seu colesterol com base na sua idade e atividade física (semanal)</h1>", unsafe_allow_html=True)
st.divider()

# Labels estilizados para os inputs
st.markdown("<p class='input-label'>Digite sua idade:</p>", unsafe_allow_html=True)
idade = st.number_input("", min_value=0, max_value=120)

st.markdown("<p class='input-label'>Digite suas horas de atividade física por semana:</p>", unsafe_allow_html=True)
atividade = st.number_input("", min_value=0, max_value=168)

# Realizando a previsão
if idade and atividade:
    colesterol_previsto = modelo.predict([[idade, atividade]])[0]  # Previsão baseada na idade e atividade
    st.write(f"O valor previsto do seu colesterol é: {colesterol_previsto:.2f}")
