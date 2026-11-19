# ClassSense

A camera-based classroom assistant that gives teachers an **advisory** view of student attention.
It never decides anything on its own: the cues only point to moments worth a look.

## What it does
- **Attention cues:** attentive, writing / reading, looking away, and head down
  (head down means the head has stayed down for a long time without lifting).
- **Phone detection:** a YOLO model tuned on our own test videos.
- **Recorded lessons:** analyse a whole video, then review the timeline, events and photos.
- **Teacher dashboard:** live class view, recorded video, and a class summary.
- Planned: roster verification, analytics and report pages, saving history to a database.

## How it works
camera -> YuNet (finds faces) -> head pose + MediaPipe (where the head points) -> attention states
-> FastAPI server -> WebSocket -> React dashboard

## Run it
Camera server (leave it running):

~~~
source venv/bin/activate
python3 server.py
~~~

Dashboard, in a second terminal:

~~~
cd frontend
npm install
npm run dev
~~~

Then open http://localhost:5173. Or double-click `start_classsense.command` to start both.

Recorded lesson: `python3 analyze_video.py path/to/lesson.mp4`, then open "Recorded Video".

Settings you can change when starting the server:

~~~
python3 server.py --phone-model phone_best2.pt --phone-conf 0.75
CLASSSENSE_DOWN_SECONDS=60 python3 server.py
~~~

## Project structure
~~~
ClassSense/
  server.py                  FastAPI server: camera, detection, live data, video, saved analyses
  analyze_video.py           Analyse a recorded lesson video
  run_live.py                Camera preview with coloured boxes; holds the face tracker
  start_classsense.command   Double-click to start everything
  cv_modules/                Detection and attention code
    attention.py             Attentive, writing, looking away, head down
    head_pose.py             Head direction from five face landmarks
    mp_pitch.py              Up/down head angle (MediaPipe)
    phone_detection.py       Phone detector (YOLO)
    models/                  Face detector and landmark model files
  scripts/                   Calibration, tuning and demo tools
  frontend/                  React dashboard (Vite, Tailwind, Recharts)
  models/                    Phone detection weights
  data/                      Test videos, saved analyses, calibration settings
  tests/                     Tests
~~~

## Limits
- The cues come from head direction, not eye gaze, so a student can look at the screen with their
  eyes while the head points elsewhere.
- Writing and resting look the same from head pose alone; a long time without lifting the head is
  what marks "head down".
- Tested with one person and a laptop camera so far; a real classroom needs its own calibration.
