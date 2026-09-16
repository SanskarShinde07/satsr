import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from fastapi import FastAPI
from fastapi.middleware.wsgi import WSGIMiddleware

from app.app import app as flask_app

api = FastAPI()
api.mount("/", WSGIMiddleware(flask_app))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(api, host="0.0.0.0", port=7860)