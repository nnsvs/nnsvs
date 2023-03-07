# Neutrino-compat

This directory contains code to run NNSVS models like NEUTRINO.

- `bin/NEUTRINO.py`: Run NNSVS's timing and acoustic feature prediction modesl with the same interface as NEUTRINO.exe.
- `bin/NSF.py`: Run NNSVS's vocoder to generate waveforms.


## Setup

You'll need the following packages to run the inference server:

```
pip install fastapi nest-asyncio uvicorn python-multipart utaupy
```

## Tyouseisientool (for windows)

https://github.com/sigprogramming/tyouseisientool

### Run NNSVS's inference server

At the neutrino_compat directory, run

```
uvicorn server:app --port 8002
```

Use your favourite port.

If you want to use difference machines for the inference server and Tyouseisientool, you can run the server with the following command:

```
uvicorn server:app --port 8002 --host 0.0.0.0
```

### Configure Tyouseisientool

- Please make sure to set `NEUTRINOフォルダのパス` to the path to the `neutrino_compat` directory. e.g., `C:\msys64\home\ryuichi\nnsvs\neutrino_compat`.
- Make sure to have NEUTRINO tools (`musicXMLtoLabel.exe`, `WORLD.exe`, and `NSF.exe`) in the `neutrino_compat\bin` directory.
- Put embedded python env (`python-3.8.10-embed-amd64`) to the `neutrino_compat` directory (contact @r9y9 if you need it).
- Change the rendering settings as follows

NEUTRINO:
```
python-3.8.10-embed-amd64\python.exe bin\NEUTRINO.py %FullLabel% %TimingLabel% %F0% %Mgc% %Bap% %ModelDir% -i %PhraseList% -p %PhraseNum% --use_api --url http://localhost:8002
```

NSF:
```
python-3.8.10-embed-amd64\python.exe bin\NSF.py %F0% %Mgc% %Bap% %ModelDir% %Wave% --use_api --url http://localhost:8002
```


The last step is to put your NNSVS's packed models to the `neutrino_compat\model\{モデル名}` directory.
Once the above steps are property done, you can use NNSVS models with Tyouseisientool.
