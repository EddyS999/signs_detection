import streamlit as st
import openai

st.title("Sign communicator")

# Initialisez le modèle OpenAI
if "openai_model" not in st.session_state:
    st.session_state["openai_model"] = "gpt-3.5-turbo"

# Initialisez la liste des messages
if "messages" not in st.session_state:
    st.session_state.messages = []

# Affichez les messages précédents
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Créez une disposition avec des colonnes pour aligner le champ d'entrée et le bouton côte à côte
col1, col2 = st.columns([4, 1])
with col1:
    prompt = st.text_input("What is up?")
with col2:
    submit_button = st.button("Send")

# Gérez l'entrée de l'utilisateur et le clic sur le bouton
if submit_button and prompt:
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Obtenez la réponse de l'assistant
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        for response in openai.ChatCompletion.create(
            model=st.session_state["openai_model"],
            messages=[
                {"role": m["role"], "content": m["content"]}
                for m in st.session_state.messages
            ],
            stream=True
        ):
            full_response += response.choices[0].delta.get("content", "")
            message_placeholder.markdown(full_response + " ")
        message_placeholder.markdown(full_response)
    st.session_state.messages.append(
        {"role": "assistant", "content": full_response})
