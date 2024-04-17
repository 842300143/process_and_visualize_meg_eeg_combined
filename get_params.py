import os
import platform
from pathlib import Path

if platform.system() != 'Windows':
    datadir = os.getenv('DATADIR')
    method =  format(os.getenv('METHOD'))
    combined_fwd_fname =  os.path.join(os.getenv('DATADIR') ,os.getenv('COMBINED_FWD_FNAME'))
    screenshots_dir = os.path.join(os.getenv('DATADIR') ,os.getenv('SCREENSHOTS_DIR'))
    n_calls = int(os.getenv('N_CALLS'))
    snr = float(os.getenv('SNR'))
    depth = float(os.getenv('DEPTH'))

else:
    method = 'dSPM'
    combined_fwd_fname = 'data/b1-combine-fwd.fif'
    screenshots_dir = 'F:/docker_workspace/test/image'
    n_calls = 50
    snr = 3.0
    depth = 0.8

