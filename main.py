from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.messages import HumanMessage, AIMessage, SystemMessage

# load environment variables 
load_dotenv()

# load embedding model
embedding_model = OpenAIEmbeddings()

vector_store = Chroma(
    persist_directory="chroma_db",
    embedding_function=embedding_model
)

retriever = vector_store.as_retriever(
    search_tyoe = "mmr",
    kwargs={
        "k": 4,
        "fetch_k": 10,
        "lamda_mult": 0.5
    }
)

#load chat model
chat_model = ChatOpenAI(model="gpt-5.5")

system_prompt =  """You are a helpful AI assistant.

                Use ONLY the provided context to answer the question.

                If the answer is not present in the context,
                say: "I could not find the answer in the document."
                """

messages = [
    SystemMessage(content=system_prompt)
]

prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                system_prompt
            ),
            (
                "human",
                """Context:
                    {context}

                    Question:
                    {question}
                """
            )
        ]
)

print("please type 0 to quit")

while True:
    query = input("You: ")
    if query == '0':
        break
    docs = retriever.invoke(query)
    
    context = "\n\n".join(
        [doc.page_content for doc in docs]
    )
    
    final_prompt = prompt.invoke({
        "context":context,
        "question":query
    })
    
    messages.append(final_prompt.to_messages()[1])
    response = chat_model.invoke(final_prompt)
    messages.append(AIMessage(content=response.content))
    print(f"AI: {response.content}")
    
print(messages)
    
