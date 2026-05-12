import asyncio                          # Provides async/await support for handling concurrent WebSocket connections
import base64                           # Decodes base64-encoded JPEG frames sent from the browser
import json                             # Parses and serialises JSON messages over WebSocket
import threading                        # Used to open the browser in a background thread without blocking the server
import webbrowser                       # Opens the default browser automatically when the server starts
from concurrent.futures import ThreadPoolExecutor  # Runs CPU-heavy face/audio analysis in a thread pool so the async loop stays responsive

import cv2                              # Decodes compressed JPEG bytes into a NumPy image array (BGR format)
import numpy as np                      # Creates NumPy arrays from raw audio samples sent by the browser
from fastapi import FastAPI, WebSocket, WebSocketDisconnect  # FastAPI is the web framework; WebSocket handles real-time bi-directional comms
from fastapi.responses import FileResponse   # Serves the index.html file when the user visits the root URL
from fastapi.staticfiles import StaticFiles  # Serves all files in the static/ folder (HTML, CSS, JS) at the /static path

from face_analyzer import FaceAnalyzer  # Python class that runs MediaPipe face landmark + emotion detection
from vocal_analyzer import VocalAnalyzer  # Python class that analyses pitch, energy, and tremor from audio samples
from trust_engine import TrustEngine    # Python class that combines face/vocal scores into a smoothed trust percentage

app = FastAPI(title="Trust Level Dashboard")  # Creates the FastAPI application instance with a human-readable title
executor = ThreadPoolExecutor(max_workers=4)  # Thread pool with 4 workers for running blocking analysis code off the async event loop

app.mount("/static", StaticFiles(directory="static"), name="static")  # Maps all requests to /static/* to files inside the static/ folder


@app.get("/")                           # Registers the root URL ("/") as an HTTP GET route
async def root():
    return FileResponse("static/index.html")  # Returns the dashboard HTML page when the user opens the server in their browser


@app.websocket("/ws")                   # Registers the WebSocket endpoint that the browser connects to for real-time analysis
async def ws_endpoint(websocket: WebSocket):
    await websocket.accept()            # Completes the WebSocket handshake so the connection is established

    face  = FaceAnalyzer()              # Creates a fresh FaceAnalyzer instance per client (each connection gets its own blink/gaze state)
    vocal = VocalAnalyzer()             # Creates a fresh VocalAnalyzer instance per client (each connection gets its own pitch history)
    trust = TrustEngine()               # Creates a fresh TrustEngine instance per client (each connection gets its own smoothed scores)
    loop  = asyncio.get_event_loop()    # Gets the running async event loop so we can submit blocking tasks to the thread pool

    print("Client connected")           # Logs to the terminal when a browser tab connects
    try:
        while True:                     # Keeps the connection open, processing messages indefinitely until the client disconnects
            raw = await websocket.receive_text()   # Waits for the next JSON message from the browser (either a video frame or audio chunk)
            msg = json.loads(raw)                  # Parses the raw JSON string into a Python dict

            if msg["t"] == "frame":                # Checks if the message is a video frame (type "frame")
                jpg   = base64.b64decode(msg["d"]) # Decodes the base64 string back into raw JPEG bytes
                arr   = np.frombuffer(jpg, np.uint8)  # Wraps the raw bytes in a NumPy array so OpenCV can read them
                frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)  # Decompresses the JPEG into a full BGR image array
                await loop.run_in_executor(executor, face.analyze, frame)  # Runs face detection in a thread pool thread so it doesn't block incoming messages

            elif msg["t"] == "audio":              # Checks if the message is an audio chunk (type "audio")
                samples = np.array(msg["d"], dtype=np.float32)  # Converts the list of float samples into a NumPy array for signal processing
                sr      = int(msg.get("sr", 44100))             # Reads the sample rate sent by the browser (defaults to 44100 Hz if missing)
                await loop.run_in_executor(executor, lambda: vocal.analyze(samples, sr))  # Runs vocal analysis in a thread pool thread

            scores = trust.update(face.last_result, vocal.last_result)  # Recalculates the smoothed trust scores using the latest face and vocal data
            label  = TrustEngine.trust_label(scores["total"])           # Converts the numeric score into a human-readable label and hex colour

            await websocket.send_text(json.dumps({   # Sends the analysis results back to the browser as a JSON string
                "scores":  scores,                   # Dict with total, facial, vocal, gaze scores (0–100 each)
                "label":   label,                    # Dict with "text" (e.g. "High Trust") and "color" (hex string)
                "metrics": {                         # Detailed per-channel metrics for the dashboard panels
                    "face":  face.last_result,       # Full face data: expressions, eye AR, blink rate, gaze deviation, landmark coords
                    "vocal": vocal.last_result,      # Full vocal data: is_speaking, pitch_stability, energy_level, tremor_index
                },
            }))

    except WebSocketDisconnect:         # Raised when the browser tab closes or navigates away
        print("Client disconnected")    # Logs the disconnection to the terminal
    except Exception as e:              # Catches any unexpected errors during the session
        print(f"WebSocket error: {e}")  # Logs the error so it can be debugged without crashing the server


if __name__ == "__main__":             # Only runs the block below when the file is executed directly (not imported)
    import uvicorn                     # Imports uvicorn here to keep it optional when the file is imported as a module
    threading.Timer(                   # Creates a one-shot timer that fires after the server has had time to start
        1.5,                           # Waits 1.5 seconds before opening the browser so uvicorn is ready to serve requests
        lambda: webbrowser.open("http://localhost:8000")  # Opens the dashboard in the default browser automatically
    ).start()                          # Starts the timer in a background thread so it doesn't block the server startup
    uvicorn.run(                       # Starts the ASGI web server
        "main:app",                    # Tells uvicorn to load the "app" object from this file ("main" module)
        host="0.0.0.0",               # Listens on all network interfaces (0.0.0.0) so it works on localhost and LAN
        port=8000,                     # Serves the dashboard on port 8000 (http://localhost:8000)
        reload=True,                   # Automatically restarts the server whenever a .py file is saved (useful during development)
    )
