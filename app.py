import streamlit as st
from streamlit_message import message
from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import (ConversationBufferMemory, 
                                                  ConversationSummaryMemory, 
                                                  ConversationBufferWindowMemory
               
                                                  )

import os
# from dotenv import load_dotenv
from config import OPENAI_API_KEY, PINECONE_API_KEY, PINECONE_API_ENVIRONMENT
from constants import PINECONE_INDEX_NAME
from langchain.chat_models import ChatOpenAI
from langchain.vectorstores import Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains.question_answering import load_qa_chain
import pinecone



# Set API key for OpenAI and Pinecone
os.environ["OPENAI_API_KEY"] = "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"
pinecone.init(
    api_key="44350c8b-72f9-4e4a-b773-4b198e3a47cd",
    environment='gcp-starter'
)

if 'conversation' not in st.session_state:
    st.session_state['conversation'] =None
if 'messages' not in st.session_state:
    st.session_state['messages'] =[]
# if 'API_Key' not in st.session_state:
#     st.session_state['API_Key'] =''

# Setting page title and header
st.set_page_config(page_title="Chat GPT Clone", page_icon=":robot_face:")
st.markdown("<h1 style='text-align: center;'>How can I assist you? </h1>", unsafe_allow_html=True)


st.sidebar.title("😎")
summarise_button = st.sidebar.button("Summarise the conversation", key="summarise")
if summarise_button:
    summarise_placeholder = st.sidebar.write("Nice chatting with you my friend ❤️:\n\n"+st.session_state['conversation'].memory.buffer)
    # summarise_placeholder.write("Nice chatting with you my friend ❤️:\n\n"+st.session_state['conversation'].memory.buffer)

#import os
#os.environ["OPENAI_API_KEY"] = "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"

def getresponse(usr_input):

    if st.session_state['conversation'] is None:

     
    # Set OpenAI LLM and embeddings
        llm_chat = ChatOpenAI(temperature=0.9, max_tokens=150,
                            model='gpt-3.5-turbo-0613', client='')
        
        st.session_state['conversation'] = ConversationChain(
            llm=llm_chat,
            verbose=True,
            memory=ConversationSummaryMemory(llm=llm_chat)
        )

        st.session_state['conversation'].predict(input=usr_input)
        print(st.session_state['conversation'].memory.buffer)

        embeddings = OpenAIEmbeddings(client='')

        # Set Pinecone index
        docsearch = Pinecone.from_existing_index(
            index_name=PINECONE_INDEX_NAME, embedding=embeddings)

        # Create chain
        chain = load_qa_chain(llm_chat)
        search = docsearch.similarity_search(usr_input)
        response = chain.run(input_documents=search, question=usr_input)

    return response, search



response_container = st.container()
# Here we will have a container for user input text box
container = st.container()


with container:
    with st.form(key='my_form', clear_on_submit=True):
        user_input = st.text_area("Your question goes here:", key='input', height=20)
        submit_button = st.form_submit_button(label='Send')

        if submit_button:
            st.session_state['messages'].append(user_input)
            model_response,model_search=getresponse(user_input)
            st.session_state['messages'].append(model_response)

            #this is the referecne fromt the embeddings
            

            with response_container:
                for i in range(len(st.session_state['messages'])):
                        if (i % 2) == 0:
                            message(st.session_state['messages'][i], is_user=True, key=str(i) + '_user')
                        else:
                            message(st.session_state['messages'][i], key=str(i) + '_AI')
        
# with st.expander('Document Similarity Search'):
     
#     # Display results
#      # search = docsearch.similarity_search(user_input)
#      print('Search results:', model_search)
#      st.write(model_search) 
                      
            
                
