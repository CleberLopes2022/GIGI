import streamlit as st
import json
import random
import base64
from sentence_transformers import SentenceTransformer, util
import torch

# Configuração do modelo
modelo = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

# Carregar base de conhecimento uma única vez
@st.cache_data
def carregar_base():
    with open("base_conhecimento.json", "r", encoding="utf-8") as file:
        return json.load(file)

base_conhecimento = carregar_base()

# Respostas padrão
respostas_padrao = [
    "Hmm... não entendi muito bem 🤔. Pode tentar reformular?",
    "Desculpe, não consegui compreender 🧠. Pode dizer de outro jeito?",
    "Acho que não peguei isso direito 😅. Pode explicar novamente?"
]

# Intenções e respostas
intencoes = {
    "saudacao": ["oi", "olá", "bom dia", "boa tarde", "boa noite", "e aí"],
    "despedida": ["tchau", "até logo", "até mais", "encerrar", "falou"],
    "agradecimento": ["obrigado", "obrigada", "valeu", "agradecido", "grato"],
    "ajuda": ["ajuda", "como funciona", "como usar", "o que você faz", "explica"]
}

respostas_intencao = {
    "saudacao": ["Olá! Que bom te ver por aqui! 😊", "Oi! Como posso te ajudar hoje?", "E aí! Pronta pra te ajudar!"],
    "despedida": ["Até mais! Se cuida. 😉", "Tchauzinho! Sempre por aqui.", "Foi ótimo falar com você. 👍"],
    "agradecimento": ["De nada! Sempre por aqui.", "Imagina! GIGI sempre pronta pra ajudar.", "Fico feliz em ajudar! 💜"],
    "ajuda": ["Posso responder perguntas! É só digitar.", "Sou uma assistente virtual treinada para te ajudar.", "Me pergunte algo e eu tentarei ajudar!"]
}

# Função para detectar intenção melhorada
def detectar_intencao(pergunta):
    pergunta_embedding = modelo.encode(pergunta.lower(), convert_to_tensor=True)
    melhor_intencao = None
    maior_similaridade = 0.5  # Definindo um limiar mínimo

    for intencao, palavras in intencoes.items():
        palavras_embedding = modelo.encode(" ".join(palavras), convert_to_tensor=True)
        similaridade = util.pytorch_cos_sim(pergunta_embedding, palavras_embedding).item()

        if similaridade > maior_similaridade:
            maior_similaridade = similaridade
            melhor_intencao = intencao

    return melhor_intencao

# Personalização de resposta com emojis
def personalizar_resposta(texto):
    return f"{texto} {random.choice(['😊', '😉', '👍', '💬', '🌟'])}"

# Buscar resposta com otimização
def encontrar_resposta(pergunta):
    intencao = detectar_intencao(pergunta)

    if intencao:
        return random.choice(respostas_intencao[intencao])

    pergunta_embedding = modelo.encode(pergunta, convert_to_tensor=True)
    melhor_resposta = random.choice(respostas_padrao)
    maior_similaridade = 0.4  # Limite mínimo de similaridade

    for chave in base_conhecimento.keys():
        chave_embedding = modelo.encode(chave, convert_to_tensor=True)
        similaridade = util.pytorch_cos_sim(pergunta_embedding, chave_embedding).item()

        if similaridade > maior_similaridade:
            maior_similaridade = similaridade
            melhor_resposta = personalizar_resposta(base_conhecimento[chave])

    return melhor_resposta

# Configuração Streamlit
st.set_page_config(page_title="GIGI - Assistente Virtual", page_icon="🤖")

# Sidebar - Exibir imagem
def imagem_em_base64(caminho):
    with open(caminho, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

img_base64 = imagem_em_base64("GIGI.jpg")

with st.sidebar:
    st.markdown(f"""
        <div style="text-align: center;">
            <img src="data:image/jpeg;base64,{img_base64}" style="width: 200px; height: 200px; border-radius: 50%;" />
            <p><b>Sou a GIGI, sua assistente virtual!</b></p>
        </div>
    """, unsafe_allow_html=True)

# Título principal
st.markdown("<h1 style='text-align: center;'>GIGI - Sua Assistente Virtual</h1>", unsafe_allow_html=True)

# Histórico otimizado
if "historico" not in st.session_state:
    st.session_state.historico = [("GIGI", "Olá! Eu sou a GIGI. Como posso te ajudar hoje?")]

# Exibição de histórico otimizada
for remetente, mensagem in st.session_state.historico[-10:]:  # Mantendo apenas as últimas 10 interações
    if remetente == "Você":
        st.chat_message("user").write(mensagem)
    else:
        st.chat_message("assistant").write(mensagem)

# Campo de entrada
with st.form(key="chat_form"):
    user_input = st.text_input("Você:", placeholder="Digite sua pergunta...")
    enviar = st.form_submit_button("Enviar")

# Processar entrada
if enviar and user_input.strip():
    with st.spinner("GIGI está pensando... 🤖💭"):
        resposta = encontrar_resposta(user_input)
        st.session_state.historico.append(("Você", user_input))
        st.session_state.historico.append(("GIGI", resposta))

    st.experimental_rerun()

# Botão de encerramento
if st.button("Encerrar conversa"):
    st.session_state.historico = [("GIGI", "Conversa encerrada. Sempre por aqui quando precisar! 💜")]
    st.experimental_rerun()

