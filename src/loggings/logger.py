import logging
import os
from datetime import datetime

# Generate filename with timestamp
LOG_FILE = f"{datetime.now().strftime('%m_%d_%Y__%H_%M_%S')}.log" # Eg: 02_16_2024__14:30_00.log

# Ensure logs folder in the root
logs_path = os.path.join(os.getcwd(), "logs")
os.makedirs(logs_path, exist_ok=True)

# Full log file path
log_file_path = os.path.join(logs_path, LOG_FILE)

# Configure logging
logging.basicConfig(
    filename=log_file_path,
    format="[%(asctime)s] - [%(filename)s:%(lineno)d] - [%(levelname)s] - %(message)s",
    level=logging.INFO
)

# Optional exposure for usage like logger.logger.info()
logger = logging