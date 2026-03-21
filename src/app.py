"""Streamlit entrypoint."""

from src.client.ui import run_client_app


def main() -> None:
    run_client_app()


if __name__ == "__main__":
    main()
