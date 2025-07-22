# web_stream.py
import json
from fastapi import FastAPI, WebSocket
from fastapi.staticfiles import StaticFiles
from event_bus import BUS

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    async for evt in BUS.subscribe():
        await ws.send_text(json.dumps(evt))
