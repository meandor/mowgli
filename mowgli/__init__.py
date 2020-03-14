import logging
import os

logging.basicConfig(
    format="%(asctime)s %(levelname)s	%(message)s "
           "[%(process)d] %(module)s %(filename)s %(funcName)s",
    level=os.environ.get("LOGLEVEL", "DEBUG")
)
