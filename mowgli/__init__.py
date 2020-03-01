import logging
import os

logging.basicConfig(
    format="%(asctime)s %(levelname)s	[%(process)d] %(module)s %(filename)s %(funcName)s %(message)s",
    level=os.environ.get("LOGLEVEL", "DEBUG")
)
