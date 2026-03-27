import sys
import os
sys.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from openenv.core.env_server import create_fastapi_app
from models import ServeAction, ServeObservation
from server.environment import LLMServeEnvironment

env = LLMServeEnvironment()

app = create_fastapi_app(env, ServeAction, ServeObservation)