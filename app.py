import streamlit as st
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from transformers import BlipProcessor, BlipForConditionalGeneration
from dotenv import load_dotenv
from PIL import Image
import torch
import os

load_dotenv()

@st.cache_resource
def load_models():
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    blip_model.to(device)
    llm = ChatGroq(model="llama-3.1-8b-instant", api_key=os.getenv("GROQ_API_KEY"))
    return processor, blip_model, device, llm

processor, blip_model, device, llm = load_models()

def generate_caption(image):
    inputs = processor(images=image, return_tensors="pt").to(device)
    out = blip_model.generate(**inputs)
    caption = processor.decode(out[0], skip_special_tokens=True)
    return caption

def process_caption_with_llm(caption):
    prompt_template = PromptTemplate(
        input_variables=["caption"],
        template="""Write a Short 40-50 word caption for the image to Post it On INSTAGRAM. If any Poster is provided, Advertise it accordingly in FIRST PERSON. 
        If any image with people or persons is provided, Write it in a first person caption, for example, 
        if a Man and a Car is in the photo, write the caption which seems to be written by the man himself for his car. 
        Also, add 5-10 LowerCase hashtags at the end, so that the image gets viral: '{caption}'. 
        Makesure to Output ONLY the CAPTION and the HASHTAGS nothing else, no title or semi title"""
    )
    llm_chain = LLMChain(llm=llm, prompt=prompt_template)
    return llm_chain.run(caption=caption)

st.title("Instagram Caption Generator")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, width=300)
    
    if st.button("Generate Caption"):
        original_caption = generate_caption(image)
        instagram_caption = process_caption_with_llm(original_caption)
        st.session_state.current_caption = instagram_caption
        st.session_state.original_caption = original_caption

if "current_caption" in st.session_state:
    st.write("Caption:")
    st.write(st.session_state.current_caption)
    
    st.write("Any Changes?")
    
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])
    
    if prompt := st.chat_input("Do you Need any Modifications?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)
        
        modify_prompt = f"""
        Current Instagram caption: {st.session_state.current_caption}
        
        User request: {prompt}
        
        Please modify the Instagram caption based on the user's request. Keep it Instagram-ready with hashtags.
        Return ONLY the modified caption.
        """
        
        llm_chain = LLMChain(
            llm=llm,
            prompt=PromptTemplate(input_variables=["prompt"], template="{prompt}")
        )
        response = llm_chain.run(prompt=modify_prompt)
        
        st.session_state.current_caption = response
        st.session_state.messages.append({"role": "assistant", "content": response})
        
        with st.chat_message("assistant"):
            st.write(response)
        
        st.rerun()