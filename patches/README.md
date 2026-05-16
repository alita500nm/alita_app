# SDK Patches

## webrtc_client_gstreamer.py
Adds queue element to audio chain in `_webrtcsrc_pad_added_cb`.
Fixes: audio dies after ~500ms when camera is active.
Apply to: `.venv/lib/python3.12/site-packages/reachy_mini/media/webrtc_client_gstreamer.py`
Date: 2026-05-16
