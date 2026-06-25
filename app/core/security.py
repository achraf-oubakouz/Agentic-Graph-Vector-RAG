from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI


def add_cors(app: FastAPI) -> None:
    """
    Attach CORS middleware to the FastAPI app.

    allow_origins=["*"]  → accepts requests from any origin.
    When you build the React frontend later, replace "*" with
    your frontend URL, e.g. ["http://localhost:5173"].
    """
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )