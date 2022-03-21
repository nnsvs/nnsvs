# NNSVS Streamlit demo
This folder explain how to work with this demo

![](streamlit_demo/imgs/demo.png)

* install the requirements
```
pip install streamlit git+https://github.com/r9y9/nnsvs
# or if you must specify the python version
python3.8 -m pip install streamlit git+https://github.com/r9y9/nnsvs
```
* download the trained models
```
[ ! -e 20220321_kiritan_timelag_mdn_duration_mdn_acoustic_resf0conv ] && curl -q -LO https://www.dropbox.com/s/87rqto5l5rpav2n/20220321_kiritan_timelag_mdn_duration_mdn_acoustic_resf0conv.zip
unzip -qq -o 20220321_kiritan_timelag_mdn_duration_mdn_acoustic_resf0conv.zip

model_dir = "20220322_yoko_timelag_mdn_duration_mdn_acoustic_resf0conv"

! [ ! -e 20220322_yoko_timelag_mdn_duration_mdn_acoustic_resf0conv ] && curl -q -LO https://www.dropbox.com/s/l1wo9dewfuk3s1v/20220322_yoko_timelag_mdn_duration_mdn_acoustic_resf0conv.zip
! unzip -qq -o 20220322_yoko_timelag_mdn_duration_mdn_acoustic_resf0conv.zip
```
* run streaimlit app
```
streamlit run streamlit_demo/app.py
# or if you must specify the python version
python3.8 -m streamlit run streamlit_demo/app.py
```
* go on localhost:8501 and load .xml file