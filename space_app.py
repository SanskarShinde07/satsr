import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import spaces
from fastapi import FastAPI
from fastapi.middleware.wsgi import WSGIMiddleware

from app.app import app as flask_app


@spaces.GPU
def _unused():
    """Declared so ZeroGPU passes its startup check. Never called."""
    return None


api = FastAPI()
api.mount("/", WSGIMiddleware(flask_app))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(api, host="0.0.0.0", port=7860)