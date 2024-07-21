import streamlit as st
from streamlit_chat import message
from qa import embed_docs, AskMe
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage


st.set_page_config(page_title="Ask Me", page_icon="🤖")
st.title("Ask Me")
with st.sidebar:
    st.header("Settings")
    fn_url = st.text_input("*****NOT USED****path of reference docs")
    embed_button = st.button("Embed")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
    AIMessage(content="Hello, I'm your Document bot. How can I help you?"),
    ]
    
#embed reference docs to vector store database
if embed_button:
    embed_docs()

a = AskMe()
user_query = st.chat_input("Type your question here...")
if user_query is not None and user_query != "":
    response = a.ask(user_query)['answer']
    print('-----------------------------------------')
    st.session_state.chat_history.append(HumanMessage(content=user_query))
    st.session_state.chat_history.append(AIMessage(content=response))

for message in st.session_state.chat_history:
    if isinstance(message, AIMessage):
        with st.chat_message("AI"):
            st.write(message.content)
    elif isinstance(message, HumanMessage):
        with st.chat_message("Human"):
            st.write(message.content)

# for n in st.session_state.chat_history:
#     print(n)

print(st.session_state)