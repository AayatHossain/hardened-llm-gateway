"""Start the gateway locally: `uv run python -m hardened_llm_gateway`."""

import uvicorn


def main() -> None:
    # Bind to localhost only until the gateway is deployed behind HTTPS.
    uvicorn.run(
        "hardened_llm_gateway.app:create_app",
        factory=True,
        host="127.0.0.1",
        port=8000,
    )


if __name__ == "__main__":
    main()
