from pathlib import Path
import logging
import sys
import os
from .server import serve


def get_ieda_path() -> Path:
    ieda_path = os.getenv("iEDA")
    if not ieda_path:
        raise EnvironmentError("Environment variable 'iEDA' is not set.")
    return Path(ieda_path)


def main(iEDA: Path | None = None, verbose: int = 2) -> None:
    import asyncio

    logging_level = logging.WARN
    if verbose == 1:
        logging_level = logging.INFO
    elif verbose >= 2:
        logging_level = logging.DEBUG

    logging.basicConfig(
        level=logging_level,
        stream=sys.stderr,
        format="%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s",
    )

    if iEDA is None:
        iEDA = get_ieda_path()

        logging.info(f"iEDA path: {iEDA}")

    serve(iEDA)


if __name__ == "__main__":
    try:
        ieda_path = get_ieda_path()
        print(f"iEDA path: {ieda_path}")
    except EnvironmentError as e:
        ieda_path = None
        print(e)

    main(ieda_path, verbose=2)
