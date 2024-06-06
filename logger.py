from datetime import datetime
import logging
import sys
import os

# Logging Section
log_file_name = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"
log_path = os.path.join(os.getcwd(), "logs")
os.makedirs(log_path, exist_ok=True)

log_files = os.path.join(log_path, log_file_name)

logging.basicConfig(
    filename=log_files,
    level=logging.INFO,
    format="[%(asctime)s] %(lineno)d %(name)s - %(levelname)s - %(message)s",
)

if __name__ == "__main__":
    logging.info("we are here")
