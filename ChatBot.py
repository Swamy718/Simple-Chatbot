from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import streamlit as st
import os

# from dotenv import load_dotenv
# load_dotenv()


#Langsmith Tracing
# os.environ["LANGSMITH_API_KEY"] = os.getenv("LANGSMITH_API_KEY")
os.environ["LANGSMITH_API_KEY"] = st.secrets["LANGSMITH_API_KEY"]
os.environ["LANGSMITH_PROJECT"] = "Simple Q&A Chatbot with Ollama"
os.environ['LANGSMITH_TRACKING']="true"
os.environ['GROQ_API_KEY']=st.secrets['GROQ_API_KEY']


#Prompt Template
prompt=ChatPromptTemplate.from_messages(
    [
        ("system","You are a helpful assistant. Please respond to the user question"),
        ("user","Question:{question}")
    ]
)

def generate_response(question,model,temp,max_tokens):
    llm=ChatGroq(model=model,temperature=temp,max_tokens=max_tokens)
    output_parser=StrOutputParser()
    chain=prompt|llm|output_parser
    answer=chain.invoke({"question":question})
    return answer


st.title("Enhanced Q&A Chatbot with Groq")


#Sidebar for settings
st.sidebar.title("Settings")

#Select the Model
model=st.sidebar.selectbox("Select the model",['Gemma2-9b-It','Llama3-8b-8192','Llama3-70b-8192'])


#Adjust response parameters
temp=st.sidebar.slider("Temperature",min_value=0.0,max_value=1.0,value=0.7)
max_tokens=st.sidebar.slider("Max Tokens",min_value=50,max_value=1000,value=300)

if "messages1" not in st.session_state:
    st.session_state.messages1=[
        {'role':'assistant','content':'Hi, I am a chatbot. How can i help you?'}
    ]

for msg in st.session_state.messages1:
    st.chat_message(msg['role']).write(msg['content'])

if input:=st.chat_input(placeholder='Ask your Question'):
    st.session_state.messages1.append({'role':'user','content':input})
    st.chat_message('user').write(input)
    response=generate_response(input,model,temp,max_tokens)
    st.session_state.messages1.append({'role':'assistant','content':response})
    st.chat_message('assistant').write(response)
