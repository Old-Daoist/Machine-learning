import streamlit as st
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama


st.set_page_config(
    page_title="El Laminator | Local study companion",
    page_icon="assets/el-laminator-mark.svg"
)

st.logo(
    "assets/el-laminator-wordmark.svg",
    size="large",
    icon_image="assets/el-laminator-mark.svg",
)


# Local Ollama model
llm = ChatOllama(
    model="llama3.2",
    temperature=0
)


# Prompt, conversation history, model, and output parser form the chain.
prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are an AI assistant, not a human. Be friendly, conversational, and helpful.
Keep answers concise by default, and explain concepts clearly when needed. Use relevant
information from the current conversation. Ask a natural follow-up question when it would
help. Be honest when you do not know something, and do not invent facts and totally no to going to a different topic unless asked or required.Don`t hide information too"""
    ),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{question}")
])

chain = prompt | llm | StrOutputParser()


st.session_state.setdefault("messages", [])

with st.sidebar:
    st.subheader("Chat settings")
    st.markdown("**Model**  \n`llama3.2`")
    st.markdown("**Backend**  \nOllama")
    st.markdown("**Framework**  \nLangChain")
    st.divider()

    if st.button(
        "Clear conversation",
        icon=":material/delete_sweep:",
        width="stretch",
    ):
        st.session_state.messages = []
        st.rerun()

st.title("El Laminator")
st.caption("A local study companion for working through questions and ideas.")
st.caption("AI can make mistakes. Check important information with reliable sources.")

# Display conversation turns retained across Streamlit reruns.
for message in st.session_state.messages:
    avatar = "assets/el-laminator-mark.svg" if message["role"] == "assistant" else None
    with st.chat_message(message["role"], avatar=avatar):
        st.markdown(message["content"])


# Handle a new turn, passing earlier turns to the prompt's history placeholder.
user_question = st.chat_input("Ask a question...")

if user_question:
    history = []
    for message in st.session_state.messages:
        if message["role"] == "user":
            history.append(HumanMessage(content=message["content"]))
        else:
            history.append(AIMessage(content=message["content"]))

    # Save the user's turn immediately so it survives reruns, including errors.
    st.session_state.messages.append({
        "role": "user",
        "content": user_question
    })

    with st.chat_message("user"):
        st.markdown(user_question)

    try:
        with st.chat_message("assistant", avatar="assets/el-laminator-mark.svg"):
            with st.spinner("Connecting to Skynet...", show_time=True):
                response_text = st.write_stream(chain.stream({
                    "history": history,
                    "question": user_question
                }))
    except Exception:
        st.error(
            "Unable to connect to Ollama. Please make sure the Ollama server is "
            "running and that the llama3.2 model is available."
        )
    else:
        st.session_state.messages.append({
            "role": "assistant",
            "content": response_text
        })