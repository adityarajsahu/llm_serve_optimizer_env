import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from openenv.core.env_server import create_fastapi_app
from models import ServeAction, ServeObservation
from server.environment import LLMServeEnvironment

app = create_fastapi_app(LLMServeEnvironment, ServeAction, ServeObservation)

def main():
    import uvicorn
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860, reload=True)

if __name__ == "__main__":
    main()