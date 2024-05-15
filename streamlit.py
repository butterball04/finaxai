from bot import co, Vectorstore, doc
import streamlit as st
import uuid

st.header("Finax AI 💬 📚")


@st.cache_resource(show_spinner=False)
def load_data():
    with st.spinner(text="Loading and indexing the docs. Hang tight! This should take 1-2 minutes."):
        vectorstore = Vectorstore(doc)
        return vectorstore


class Chatbot:
    def __init__(self, vectorstore: Vectorstore):
        self.vectorstore = vectorstore
        self.conversation_id = str(uuid.uuid4())
        self.citations = []
        self.cited_documents = []

    def run(self, message: str):
        # Generate search queries (if any)
        response = co.chat(message=message,
                           model="command-r",
                           search_queries_only=True)

        # If there are search queries, retrieve document chunks and respond
        if response.search_queries:
            print("Retrieving information...", end="")

            # Retrieve document chunks for each query
            documents = []
            for query in response.search_queries:
                documents.extend(self.vectorstore.retrieve(query.text))

            # Use document chunks to respond
            response = co.chat_stream(
                message=message,
                model="command-r-plus",
                documents=documents,
                conversation_id=self.conversation_id,
            )
        # If there is no search query, directly respond
        else:
            response = co.chat_stream(
                message=message,
                model="command-r-plus",
                conversation_id=self.conversation_id,
            )

        # Display response
        for event in response:
            if event.event_type == "text-generation":
                yield str(event.text)
            # elif event.event_type == "citation-generation":
            #     self.citations.extend(event.citations)
            # elif event.event_type == "stream-end":
            #     self.cited_documents = event.response.documents


if "messages" not in st.session_state.keys():  # Initialize the chat message history
    st.session_state.messages = [
        {"role": "assistant",
            "content": "Ask me a question about Mercari's 12th Period 3rd Quarter(2024.01.01-2024.03.31)"}
    ]


# Prompt for user input and save to chat history
if prompt := st.chat_input("Your question"):
    st.session_state.messages.append({"role": "user", "content": prompt})

for message in st.session_state.messages:  # Display the prior chat messages
    with st.chat_message(message["role"]):
        st.write(message["content"])


# If last message is not from assistant, generate a new response
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            vectorstore = load_data()
            chatbot = Chatbot(vectorstore=vectorstore)
            response = chatbot.run(prompt)
            full_response = st.write_stream(response)
            message = {"role": "assistant", "content": full_response}
            st.session_state.messages.append(message)
