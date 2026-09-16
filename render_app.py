import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from app.app import app

if __name__ == "__main__":
    import os
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 10000)))