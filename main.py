
import streamlit as st
# To get environment variables
import os
from datetime import date
from io import StringIO





# Make the display a bit wider
##from IPython.display import display, HTML
##display(HTML("<style>.container { width:90% !important; }</style>"))

# To split our transcript into pieces
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Our chat model. We'll use the default which is gpt-3.5-turbo
from langchain.chat_models import ChatOpenAI
from langchain.chains.summarize import load_summarize_chain

# Prompt templates for dynamic values
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate, # I included this one so you know you'll have it but we won't be using it
    HumanMessagePromptTemplate
)

# To create our chat messages
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)
import langchain

from langchain import PromptTemplate
from langchain.llms import OpenAI



def load_LLM(input_patient_name, input_situation, stringio):
    os.environ['OPENAI_API_KEY'] = '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'
    ##with open('CBTmodel.txt', 'r') as file:
    content = stringio.read()

    ##print(content[:215]) # Why 215? Because it cut off at a clean line

    text_splitter = RecursiveCharacterTextSplitter(separators=["\n", "\n"], chunk_size=5000, chunk_overlap=250)
    texts = text_splitter.create_documents([content])

    ##print (f"You have {len(texts)} texts")
    ##texts[0]

    llm = ChatOpenAI(temperature=0)

    template="""

    You are a medical scribe for the therapist in the transcript. Given the following conversation between a patient and a therapist, analyze the dialogue and identify details strictly based on the provided text. Categorize the key details into the Subjective, Objective, Assessment, and Plan components of a SOAP note. Remember to maintain respect for the patient's narrative and focus only on the details relevant to their mental health and care:

    Do not respond with anything outside of the transcript. If you don't know, say, "I don't know" and leave area blank.
    It will be relevant to the situation {situation_decription}.

    Fill information in with the following format and leave anything blank that is not in transcript:

    Subjective: What are the patient's reported feelings, thoughts, and perceptions directly relevant to their mental health? Capture any specific quotes from the patient that illustrate their symptoms or complaints, while avoiding unnecessary details of their personal stories.

    Objective: What are the therapist's observations of the patient's behavior or demeanor? Try to identify mentions of mood, affect, cooperativeness, insight, cognitive function, eye contact, tone of voice, etc.

    Assessment: Based on the conversation, what clinical impressions or diagnoses can be inferred strictly from the conversation? Stick strictly to the information provided and avoid assumptions.

    Plan: What future actions, treatments, or plans were discussed? Include referrals, additional services, plans for the next session, or discussions of medication, if mentioned.


    """
    system_message_prompt_map = SystemMessagePromptTemplate.from_template(template)

    human_template="{text}" # Simply just pass the text as a human message
    human_message_prompt_map = HumanMessagePromptTemplate.from_template(human_template)

    chat_prompt_map = ChatPromptTemplate.from_messages(messages=[system_message_prompt_map, human_message_prompt_map])

    ##-------------------------------------------------
    template="""

    "Given the following key details from a therapy session, generate a SINGLE SOAP note following the given structure. Maintain focus on the provided information, avoid introducing new information not present in the details, and respect the patient's narrative. Include all information:

    Subjective Details:
    Objective Details:
    Assessment Details:
    Plan Details:

    Generate a SINGLE SOAP note with the following structure:

    Patient Name: {Patient_Name}
    Session Date: {Session_Date}

    SUBJECTIVE: A summary of the patient's reported feelings, thoughts, and perceptions that are directly relevant to their mental health. Include patient quotes that highlight their symptoms or complaints, while maintaining respect for their personal stories.

    OBJECTIVE: Summarize any observations made by the therapist about the patient's behavior or demeanor during the session. Include information about the patient's mood, affect, cooperativeness, insight, cognitive function, eye contact, tone of voice, etc., if provided.

    ASSESSMENT: Provide a concise interpretation of the patient's mental health status based on the conversation, giving clinical impressions or potential diagnoses. Avoid assumptions or speculations.

    PLAN: Outline the recommended next steps or treatment plan. This should include referrals or additional services, plans for the next session, and discussions of medication, if applicable."


    """
    system_message_prompt_combine = SystemMessagePromptTemplate.from_template(template)

    human_template="{text}" # Simply just pass the text as a human message
    human_message_prompt_combine = HumanMessagePromptTemplate.from_template(human_template)

    chat_prompt_combine = ChatPromptTemplate.from_messages(messages=[system_message_prompt_combine, human_message_prompt_combine])

    chain = load_summarize_chain(llm,
                                chain_type="map_reduce",
                                map_prompt=chat_prompt_map,
                                combine_prompt=chat_prompt_combine,
                                verbose=True
                                )

    output = chain.run({
                        "input_documents": texts,
                        "Patient_Name" : input_patient_name,
                        "Session_Date" : date.today(),

                        "situation_decription" : input_situation
                    })

    st.write(output)
##----------------------------------------------------------------------
st.set_page_config(page_title="Consult AI", page_icon=": 0_0 ")
st.header("Consult AI - Your very own AI Medical Scribe")

col1, col2 = st.columns(2)

with col1:
    st.markdown("Experience the revolution in healthcare documentation with our AI-powered scribing tool, transforming your patient-therapist transcripts into concise, professional SOAP notes in an instant. Save valuable time, reduce administrative load, and focus on what truly matters - patient care.")
    
with col2:
    st.image(image='frontPageImage.jpg', width=500, caption='')

st.markdown("## Enter Your transcript .txt file here:")
input_patient_name = ""
input_descrip = ""
with open('CBTmodel.txt', 'r') as file:
    file_content = file.read()
if st.button('CBT Example - Lucy'):
    input_patient_name = "Lucy"
    input_descrip = "CBT model. her first session with the therapist."
    st.download_button(
        label="Please  download this and upload below.",
        data= file_content,
        file_name='CBTmodel.txt',
        mime='text/plain',
    )
with st.form("input_form"):

    
    col1, col2 = st.columns(2)

    with col1:
        uploaded_transcript = st.file_uploader("Upload file here", accept_multiple_files=False)

    with col2:
        st.markdown("Make sure the file is a .txt format with SPEAKER 1 and SPEAKER 2 addressed as Patient or Therapist.")
        input_patient_name = st.text_input(label="Patient's name...(Leave blank if you want)", value=input_patient_name, key="patient_name_input")
        input_situation = st.text_area(label="In one scentence, describe the Session reason...", value=input_descrip, key="situation_input")

    ##def get_text():
        ##input_text = st.text_area(label="Your Email...", key="email_input")
        ##return input_text

    ##email_input = get_text()
    submitted = st.form_submit_button("Submit")

if submitted:
    stringio = StringIO(uploaded_transcript.getvalue().decode("utf-8"))
    st.write(stringio)
    st.markdown("## Your AI Generated SOAP Note:")
    final_output = load_LLM(input_patient_name, input_situation, stringio)
    st.write(final_output)
st.markdown("use Example?")






##if uploaded_transcript:
    ##prompt_with_email = prompt.format(tone=option_tone, dialect=option_dialect, email=email_input)
    ##formatted_email = llm(prompt_with_email)

##st.write(formatted_email)