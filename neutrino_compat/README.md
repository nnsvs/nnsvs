# Neutrino-compat

This directory contains code to run NNSVS models like NEUTRINO.

- `bin/NEUTRINO.py`: Run NNSVS's timing and acoustic feature prediction modesl with the same interface as NEUTRINO.exe.
- `bin/NSF.py`: Run NNSVS's vocoder to generate waveforms.


## Setup

TBD

## How to use

TBD

### Tyouseisientool

https://github.com/sigprogramming/tyouseisientool

#### Settings using NNSVS with local inference mode

NEUTRINO:
```
python-3.8.10-embed-amd64\python.exe bin\NEUTRINO.py %FullLabel% %TimingLabel% %F0% %Mgc% %Bap% %ModelDir% -i %PhraseList% -p %PhraseNum%
```

NSF:
```
python-3.8.10-embed-amd64\python.exe bin\NSF.py %F0% %Mgc% %Bap% %ModelDir% %Wave%
```

#### Settings using NNSVS's server inference

TODO: docs on running server locally or google colab

NEUTRINO:
```
python-3.8.10-embed-amd64\python.exe bin\NEUTRINO.py %FullLabel% %TimingLabel% %F0% %Mgc% %Bap% %ModelDir% -i %PhraseList% -p %PhraseNum% --use_api --url ${path_to_server_url}
```

NSF:
```
python-3.8.10-embed-amd64\python.exe bin\NSF.py %F0% %Mgc% %Bap% %ModelDir% %Wave% --use_api --url ${path_to_server_url}
```
