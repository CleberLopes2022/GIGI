import streamlit as st
import json
import spacy
import spacy.cli
import random
import base64

# Carregar NLP
nlp = spacy.load("pt_core_news_md")


# Base de conhecimento
def carregar_base():
    with open("base_conhecimento.json", "r", encoding="utf-8") as file:
        return json.load(file)

base_conhecimento = carregar_base()

# Respostas padrão da GIGI
respostas_padrao = [
    "Hmm... não entendi muito bem 🤔. Pode tentar reformular, por favor?",
    "Desculpe, não consegui compreender 🧠. Pode dizer de outro jeito?",
    "Acho que não peguei isso direito 😅. Pode explicar novamente?"
]

# Intenções conhecidas
intencoes = {
    "saudacao": ["oi", "olá", "bom dia", "boa tarde", "boa noite", "e aí"],
    "despedida": ["tchau", "até logo", "até mais", "encerrar", "falou"],
    "agradecimento": ["obrigado", "obrigada", "valeu", "agradecido", "grato"],
    "ajuda": ["ajuda", "como funciona", "como usar", "o que você faz", "explica"]
}

respostas_intencao = {
    "saudacao": [
        "Olá! Que bom te ver por aqui! ",
        "Oi, tudo bem? Como posso te ajudar hoje? ",
        "E aí! Pronta pra te ajudar com o que precisar! "
    ],
    "despedida": [
        "Até mais! Se cuida.",
        "Tchauzinho! Quando quiser conversar, estarei aqui .",
        "Foi ótimo falar com você. Até logo! "
    ],
    "agradecimento": [
        "De nada! Sempre que precisar, estou por aqui ",
        "Imagina! GIGI sempre pronta pra ajudar ",
        "Fico feliz em ajudar! "
    ],
    "ajuda": [
        "Posso responder perguntas com base na minha base de conhecimento! É só digitar.",
        "Sou uma assistente virtual treinada pra entender e responder perguntas.",
        "Me pergunte algo e eu tentarei ajudar!"
    ]
}

# Função para detectar intenção
def detectar_intencao(pergunta):
    pergunta = pergunta.lower()
    for intencao, palavras in intencoes.items():
        if any(p in pergunta for p in palavras):
            return intencao
    return None

# Resposta personalizada com emojis
def personalizar_resposta(texto):
    emojis = ["😊", "😉", "👍", "💬", "🌟"]
    return f"{texto} {random.choice(emojis)}"

# Encontrar resposta da GIGI
def encontrar_resposta(pergunta):
    intencao = detectar_intencao(pergunta)

    if intencao:
        return random.choice(respostas_intencao[intencao])

    pergunta_nlp = nlp(pergunta.lower())
    melhor_resposta = random.choice(respostas_padrao)
    maior_similaridade = 0.0

    for chave in base_conhecimento.keys():
        chave_nlp = nlp(chave.lower())
        similaridade = pergunta_nlp.similarity(chave_nlp)

        if similaridade > maior_similaridade:
            maior_similaridade = similaridade
            melhor_resposta = personalizar_resposta(base_conhecimento[chave])

    return melhor_resposta

# Streamlit config
st.set_page_config(
    page_title="GIGI - Assistente Virtual",
    page_icon="🤖",
    layout="centered"
)

# --- SIDEBAR ---

def imagem_em_base64(caminho):
    with open(caminho, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

img_base64 = imagem_em_base64("GIGI.jpg")

with st.sidebar:
    st.markdown(
        f"""
        <div style="text-align: center;">
            <img src="data:image/jpeg;base64,{img_base64}"
                 style="width: 200px; height: 200px; border-radius: 50%; object-fit: cover; border: 3px solid #ccc;" />
            <p style="margin-top: 10px; font-weight: bold;">Sou a GIGI, a assistente virtual DGI.</p>
        </div>
        """,
        unsafe_allow_html=True
    )

# --- Título ---
st.markdown("<h1 style='text-align: center; color: #f5f5fa;'> GIGI - Sua Assistente Virtual</h1>", unsafe_allow_html=True)

# Iniciar histórico
if "historico" not in st.session_state:
    st.session_state.historico = [("GIGI", "Olá! Eu sou a GIGI . Como posso te ajudar hoje?")]

# Entrada
with st.form(key="chat_form"):
    user_input = st.text_input("Você:", placeholder="Digite sua pergunta para a GIGI...", key="input_user")
    enviar = st.form_submit_button("Enviar")

# Processar pergunta
if enviar and user_input.strip() != "":
    resposta = encontrar_resposta(user_input)
    st.session_state.historico.append(("Você", user_input))
    st.session_state.historico.append(("GIGI", resposta))

# Exibir histórico
# st.markdown("## 🗨️ Conversa")
for remetente, mensagem in st.session_state.historico:
    if remetente == "Você":
        st.markdown(f"<div style='text-align: left; background-color: #0b5c11; padding: 25px; border-radius: 10px; margin: 5px;'>{mensagem}</div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div style='text-align: left; background-color: #6364a8; padding: 25px; border-radius: 10px; margin: 5px;'><strong>GIGI:</strong> {mensagem}</div>", unsafe_allow_html=True)

# Botão para encerrar
if st.button("Encerrar conversa"):
    st.session_state.historico = [("GIGI", "Conversa encerrada. Quando quiser conversar de novo, estarei por aqui! 💜")]
