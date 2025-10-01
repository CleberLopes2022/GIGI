import streamlit as st
import json
import random
import base64
from sentence_transformers import SentenceTransformer, util
import torch
import unicodedata
import re


# ---------- 1. Funções de normalização ----------
def normalizar_texto(texto: str) -> str:
    """
    • Converte para minúsculas
    • Remove acentos (Unicode NFKD → ASCII)
    • Remove pontuação e espaços extras
    """
    texto = texto.lower()
    texto = unicodedata.normalize("NFKD", texto).encode("ascii", "ignore").decode("utf-8")
    texto = re.sub(r"[^\w\s]", "", texto)          # remove pontuação
    texto = re.sub(r"\s+", " ", texto).strip()      # espaços duplos → único
    return texto


# Configuração da página - DEVE SER O PRIMEIRO COMANDO
st.set_page_config(page_title="GIGI - Assistente Virtual", page_icon="GIGI.jpg", initial_sidebar_state="expanded")

# Carregar o modelo apenas uma vez
@st.cache_resource
def carregar_modelo():
    return SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

modelo = carregar_modelo()

# ---------- 2. Carregamento da base ----------
@st.cache_data
def carregar_base():
    with open("base_conhecimento.json", "r", encoding="utf-8") as file:
        dados = json.load(file)
    # Normaliza as chaves – elas serão usadas como “perguntas” na busca
    return {normalizar_texto(k): v for k, v in dados.items()}


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

def detectar_intencao(pergunta, limiar=0.65):
    """
    Detecta a intenção do usuário comparando a pergunta com exemplos.
    Só retorna se a similaridade for >= limiar.
    """
    pergunta_embedding = modelo.encode(pergunta.lower(), convert_to_tensor=True)
    melhor_intencao = None
    maior_similaridade = 0.0  

    for intencao, exemplos in intencoes.items():
        exemplos_embedding = modelo.encode(exemplos, convert_to_tensor=True)
        similaridade = util.pytorch_cos_sim(pergunta_embedding, exemplos_embedding).max().item()

        if similaridade > maior_similaridade:
            maior_similaridade = similaridade
            melhor_intencao = intencao

    return melhor_intencao if maior_similaridade >= limiar else None

# Respostas personalizadas
def personalizar_resposta(texto):
    return f"{texto} {random.choice(['😊', '😉', '👍', '💬', '🌟'])}"


# ---------- 5. Busca de resposta ----------
def encontrar_resposta(pergunta: str) -> str:
    """
    1. Procura resposta na base JSON (prioridade)
    2. Se não encontrar, tenta identificar intenção
    3. Se nada for encontrado, usa resposta padrão
    """
    pergunta_norm = normalizar_texto(pergunta)
    pergunta_emb = modelo.encode(pergunta_norm, convert_to_tensor=True)

    melhor_resposta = None
    maior_sim = 0.6  # limiar mais flexível para achar no JSON

    # ---------- 1. Busca na base JSON ----------
    for chave, emb_chave in embeddings_base.items():
        sim = util.pytorch_cos_sim(pergunta_emb, emb_chave).item()
        if sim > maior_sim:
            maior_sim = sim
            melhor_resposta = base_conhecimento[chave]

    if melhor_resposta:
        return personalizar_resposta(melhor_resposta)

    # ---------- 2. Intenção ----------
    intencao = detectar_intencao(pergunta)
    if intencao:
        return personalizar_resposta(random.choice(respostas_intencao[intencao]))

    # ---------- 3. Resposta padrão ----------
    return personalizar_resposta(random.choice(respostas_padrao))



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
            <p style="color:#00a2ff; font-weight: bold; margin-top: 10px;">
                <b>Sou a GIGI, sua assistente virtual!</b>
            </p>
        </div>
        <div style="text-align: center; margin-bottom: 15px; color: #00a2ff">
            <p>
                Meu objetivo é facilitar o seu dia! 
                Posso responder dúvidas frequentes, fornecer informações úteis e muito mais — tudo com inteligência artificial.
            </p>
            <p>
                Experimente me perguntar algo como:
                <br>1 - Qual o email do credenciamento?
                <br>2 - Preciso do link do portal frotista?
                <br>3 - Preciso do link do formulário de Gestão de acessos.
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



































