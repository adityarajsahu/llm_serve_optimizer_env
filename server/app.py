try:
    from openenv.core.env_server.http_server import create_app
except ImportError as e:
    raise ImportError(
        "openenv is required. Install with: uv sync"
    ) from e

try:
    from ..models import ServeAction, ServeObservation
    from .environment import LLMServeEnvironment
except ImportError:
    from models import ServeAction, ServeObservation
    from server.environment import LLMServeEnvironment

app = create_app(
    LLMServeEnvironment,
    ServeAction,
    ServeObservation,
    env_name="llm_serve_optimizer_env",
    max_concurrent_envs=1,
)


def main(host: str = "0.0.0.0", port: int = 7860):
    import uvicorn
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    import argparse
    import sys
    if len(sys.argv) > 1:
        parser = argparse.ArgumentParser()
        parser.add_argument("--port", type=int, default=7860)
        args = parser.parse_args()
        main(port=args.port)
    else:
        main()