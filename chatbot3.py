import streamlit as st
import json
import random
import base64
from sentence_transformers import SentenceTransformer, util
import torch

# Configuração da página - DEVE SER O PRIMEIRO COMANDO
st.set_page_config(page_title="GIGI - Assistente Virtual", page_icon="GIGI.jpg", initial_sidebar_state="expanded")

# Carregar o modelo apenas uma vez
@st.cache_resource
def carregar_modelo():
    return SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

modelo = carregar_modelo()

# Carregar base de conhecimento com cache
@st.cache_data
def carregar_base():
    with open("base_conhecimento.json", "r", encoding="utf-8") as file:
        return json.load(file)

base_conhecimento = carregar_base()

# Pré-calcular embeddings da base
@st.cache_data
def calcular_embeddings_base():
    return {chave: modelo.encode(chave, convert_to_tensor=True) for chave in base_conhecimento.keys()}

embeddings_base = calcular_embeddings_base()


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


def detectar_intencao(pergunta):
    pergunta_embedding = modelo.encode(pergunta.lower(), convert_to_tensor=True)
    melhor_intencao = None
    maior_similaridade = max(0.5, min(0.7, len(pergunta) / 50))

    for intencao, palavras in intencoes.items():
        palavras_embedding = modelo.encode(" ".join(palavras), convert_to_tensor=True)
        similaridade = util.pytorch_cos_sim(pergunta_embedding, palavras_embedding).item()

        if similaridade > maior_similaridade:
            maior_similaridade = similaridade
            melhor_intencao = intencao

    return melhor_intencao

# Respostas personalizadas
def personalizar_resposta(texto):
    return f"{texto} {random.choice(['😊', '😉', '👍', '💬', '🌟'])}"


# Busca de resposta otimizada
def encontrar_resposta(pergunta):
    intencao = detectar_intencao(pergunta)
    if intencao:
        return random.choice(respostas_intencao[intencao])

    pergunta_embedding = modelo.encode(pergunta, convert_to_tensor=True)
    melhor_resposta = random.choice(respostas_padrao)
    maior_similaridade = 0.7


    for chave, chave_embedding in embeddings_base.items():
        similaridade = util.pytorch_cos_sim(pergunta_embedding, chave_embedding).item()
        if similaridade > maior_similaridade:
            maior_similaridade = similaridade
            melhor_resposta = f"{base_conhecimento[chave]} {random.choice(['😊', '😉', '👍', '💬', '🌟'])}"

    return melhor_resposta


# Sidebar - Exibir imagem com borda futurista
def imagem_em_base64(caminho):
    with open(caminho, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

img_base64 = imagem_em_base64("GIGI.jpg")

with st.sidebar:
    st.markdown(f"""
        <div style="text-align: center;">
            <img src="data:image/jpeg;base64,{img_base64}" 
                 style="width: 200px; height: 200px; border-radius: 50%;
                        border: 4px solid #00ffff; 
                        box-shadow: 0 0 20px #00ffff;
                        margin-bottom: 15px;" />
            <p <p style="color:#02e3e3; font-weight: bold; margin-top: 10px;"><b>Sou a GIGI, sua assistente virtual!</b></p>
        </div>
        <div style="text-align: center; margin-bottom: 15px; color: #02e3e3">
            <p>
                Meu objetivo é facilitar o seu dia! 
                Posso responder dúvidas frequentes,fornecer informações úteis e muito mais — tudo com inteligência artificial.
            </p>
            <p>
                Experimente me perguntar algo como "qual o email do credenciamento? ou preciso do link do portal frotista?.
            </p>
        </div>
    """, unsafe_allow_html=True)

# Título principal
st.markdown("<h1 style='text-align: center;'>GIGI - Sua Assistente Virtual</h1>", unsafe_allow_html=True)


if "historico" not in st.session_state:
    st.session_state.historico = [("GIGI", "Olá! Eu sou a GIGI. Como posso te ajudar hoje?")]

# Exibição do histórico de conversa antes do campo de entrada
st.markdown("Histórico da conversa")
for remetente, mensagem in st.session_state.historico[-10:]:
    if remetente == "Você":
        st.chat_message("user").write(mensagem)
    else:
        st.chat_message("assistant").write(mensagem)

# Função para processar a resposta e atualizar o histórico
def processar_pergunta():
    user_input = st.session_state.input_user.strip()
    if user_input:
        resposta = encontrar_resposta(user_input)
        st.session_state.historico.append(("Você", user_input))
        st.session_state.historico.append(("GIGI", resposta))
        st.session_state.input_user = ""  # Reset de campo sem erro

# Formulário de entrada abaixo do histórico
st.markdown("Digite sua pergunta abaixo")
with st.form(key="chat_form"):
    user_input = st.text_input("Você:", placeholder="Digite sua pergunta...", key="input_user")
    enviar = st.form_submit_button("Enviar", on_click=processar_pergunta)


if enviar and user_input.strip():
    with st.spinner("GIGI está pensando... 🤖💭"):
        resposta = encontrar_resposta(user_input)
        st.session_state.historico.append(("Você", user_input))
        st.session_state.historico.append(("GIGI", resposta))
    # Agora, resetamos corretamente
    st.session_state.input_user = ""

    st.rerun()
















