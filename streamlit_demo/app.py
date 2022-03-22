import tempfile

import librosa
import numpy as np
import pysinsy
import soundfile as sf
import streamlit as st
from nnmnkwii.io import hts
from nnsvs.pretrained import create_svs_engine
from nnsvs.svs import SPSVS

st.title("NNSVS Demo")
st.markdown("Upload your .xml music file with text as input to make it sing.")

models = {
    "kiritan": "r9y9/20220321_kiritan_timelag_mdn_duration_mdn_acoustic_resf0conv",
    "yoko": "r9y9/20220322_yoko_timelag_mdn_duration_mdn_acoustic_resf0conv",
}

voice_option = st.selectbox("Select the voice", models.keys())
uploaded_file = st.file_uploader("Choose a .xml music file", type="xml")

if st.button("synthesis") and uploaded_file:
    with st.spinner(f"Synthesizing to wav"):
        # synthesize
        contexts = pysinsy.extract_fullcontext(uploaded_file.name)
        labels = hts.HTSLabelFile.create_from_contexts(contexts)

        engine = create_svs_engine(models[voice_option])
        wav, sr = engine.svs(labels)

        wav = librosa.effects.trim(wav.astype(np.float64), top_db=40)[0].astype(
            np.int16
        )

        # show audio player
        with tempfile.NamedTemporaryFile(suffix=".wav") as f:
            sf.write(f.name, wav, sr)
            with open(f.name, "rb") as wav_file:
                st.audio(wav_file.read(), format="audio/wav")
