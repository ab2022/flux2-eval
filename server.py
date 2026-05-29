"""Entry point: launch the FLUX.2-dev evaluation server."""

import uvicorn


def main():
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        workers=1,
        log_level="info",
    )


if __name__ == "__main__":
    main()
