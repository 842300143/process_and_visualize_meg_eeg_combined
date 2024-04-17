#!/bin/sh
Xvfb :1 -screen 0 1280x1024x24 -ac &
python process_and_visualize_meg_eeg_combined.py