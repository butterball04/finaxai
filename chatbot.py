from bot import co, Vectorstore, vectorstore
import uuid


class Chatbot:
    def __init__(self, vectorstore: Vectorstore):
        self.vectorstore = vectorstore
        self.conversation_id = str(uuid.uuid4())

    def run(self):

        while True:

            # Get the user message
            message = input("User: ")

            # Typing "quit" ends the conversation
            if message.lower() == "quit":
                print("Ending chat.")
                break

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
 # Print the chatbot response, citations, and documents
            print("\nChatbot:")
            citations = []
            cited_documents = []

            # Display response
            for event in response:
                if event.event_type == "text-generation":
                    print(event.text, end="")
                elif event.event_type == "citation-generation":
                    citations.extend(event.citations)
                elif event.event_type == "stream-end":
                    cited_documents = event.response.documents

            # Display citations and source documents
            if citations:
                print("\n\nCITATIONS:")
                for citation in citations:
                    print(citation)

                print("\nDOCUMENTS:")
                for document in cited_documents:
                    print(document)

            print(f"\n{'-'*100}\n")


chatbot = Chatbot(vectorstore)

# Run the chatbot
chatbot.run()
