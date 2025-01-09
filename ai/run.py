import sys
from faseeh import FaseehProject

import logging
import warnings
warnings.simplefilter("ignore", UserWarning)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s",
    datefmt='%Y-%m-%d %H:%M:%S'
)

if __name__ == "__main__":
    # get the configuration file path from args
    config_path = sys.argv[1]
    project = FaseehProject(config_path)
    project.execute()