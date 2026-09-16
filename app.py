import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import gradio as gr
from fastapi import FastAPI
from fastapi.middleware.wsgi import WSGIMiddleware

from app.app import app as flask_app

api = FastAPI()

with gr.Blocks() as placeholder:
    gr.Markdown("SATSR")

api = gr.mount_gradio_app(api, placeholder, path="/gradio")
api.mount("/", WSGIMiddleware(flask_app))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(api, host="0.0.0.0", port=7860)