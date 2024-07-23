import numpy as np
import os
import requests

# %% Download Steinmetz dataset

fname = []
for j in range(3):
    fname.append(f'steinmetz_part{j}.npz')

url = ["https://osf.io/agvxh/download",
       "https://osf.io/uv3mw/download",
       "https://osf.io/ehmw2/download"]

for j in range(len(url)):
    if not os.path.isfile(fname[j]):
        try:
            r = requests.get(url[j])
        except requests.ConnectionError:
            print("!!! Failed to download data !!!")
        else:
            if r.status_code != requests.codes.ok:
                print("!!! Failed to download data !!!")
            else:
                with open(fname[j], "wb") as fid:
                    fid.write(r.content)

# %% Load dataset

alldat = np.array([])
for j in range(len(fname)):
    alldat = np.hstack(
        (alldat,
         np.load(f'steinmetz_part{j}.npz',
                 allow_pickle=True)['dat']))
