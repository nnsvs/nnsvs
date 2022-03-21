import os
import streamlit as st
import soundfile as sf

import numpy as np

import librosa
import pysinsy
from nnmnkwii.io import hts
from nnsvs.svs import SPSVS

st.title("NNSVS Demo")
st.markdown("Upload your .xml music file with text as input to make it singed.")

models = {
  'kiritan': '20220321_kiritan_timelag_mdn_duration_mdn_acoustic_resf0conv',
  'yoko' : '20220322_yoko_timelag_mdn_duration_mdn_acoustic_resf0conv'
}

voice_option = st.selectbox(
     'Select the voice',
     models.keys()
)
uploaded_file = st.file_uploader("Choose a .xml music file", type='xml')

if uploaded_file:
  with st.spinner(f"Sinthezizing to wav"):
    # save input file
    with open(os.path.join(os.getcwd(), uploaded_file.name), "wb") as f:
      f.write(uploaded_file.getbuffer())
    file_path = os.path.join(os.getcwd(), uploaded_file.name)

    # synthesize
    model_dir = models[voice_option]
    contexts = pysinsy.extract_fullcontext(file_path)
    labels = hts.HTSLabelFile.create_from_contexts(contexts)

    engine = SPSVS(model_dir)
    wav, sr = engine.svs(labels)

    wav = librosa.effects.trim(wav.astype(np.float64), top_db=40)[0]
    
    # save wav
    out_path = file_path + '.wav'
    sf.write(out_path, wav, sr)

    # show audio player
    with open(out_path, 'rb') as audio_file:
      audio_bytes = audio_file.read()
    st.audio(audio_bytes)

