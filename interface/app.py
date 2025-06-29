import streamlit as st
from llama_indext import ChatOllama
import shelve
import warnings
warnings.filterwarnings("ignore")

ai_assistant = ChatOllama()
st.title("Ollama Chatbot Demo")

def chatbot_response(user_input):
    output = ai_assistant.interact_with_llm(user_input)
    return output

USER_AVATAR = "👤"
BOT_AVATAR = "🤖"

# Ensure openai_model is initialized in session state
if "ollama_model" not in st.session_state:
    st.session_state["ollama_model"] = "mistral:7B-instruct"

# Load chat history from shelve file
def load_chat_history():
    with shelve.open("chat_history") as db:
        return db.get("messages", [])


# Save chat history to shelve file
def save_chat_history(messages):
    with shelve.open("chat_history") as db:
        db["messages"] = messages


# Initialize or load chat history
if "messages" not in st.session_state:
    st.session_state.messages = load_chat_history()

# Sidebar with a button to delete chat history
with st.sidebar:
    if st.button("Delete Chat History"):
        st.session_state.messages = []
        save_chat_history([])

# Display chat messages
for message in st.session_state.messages:
    avatar = USER_AVATAR if message["role"] == "user" else BOT_AVATAR
    with st.chat_message(message["role"], avatar=avatar):
        st.markdown(message["content"])

# Main chat interface
if prompt := st.chat_input("How can I help?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar=USER_AVATAR):
        st.markdown(prompt)

    with st.chat_message("assistant", avatar=BOT_AVATAR):
        message_placeholder = st.empty()
        full_response = ""
        query = st.session_state["messages"][-1]['content'].lower()
        
        print("message: ", query)
        full_response = ai_assistant.interact_with_llm(
            query
        )
        print("response: ", full_response)
            # full_response += response.choices[0].delta.content or ""
            # message_placeholder.markdown(full_response + "|")
        message_placeholder.markdown(full_response)
    st.session_state.messages.append({"role": "assistant", "content": full_response})

# Save chat history after each interaction
save_chat_history(st.session_state.messages)