import argparse
import sys
import os

# This is a bit of a hack to make sure src is in the python path
# A better way would be to install the project as a package
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


from src.pipeline.pipeline import FaceSwapPipeline
from src.loggings.logger import logger
from src.exceptions.exception import FaceDetectionException

def main(video_path: str, image_path: str):
    try:
        pipeline = FaceSwapPipeline(video_path=video_path, image_path=image_path)
        pipeline.run()
    except FaceDetectionException as e:
        logger.error(f"A controlled exception occurred: {e}")
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the Face Swapping pipeline.")
    parser.add_argument("--video", type=str, required=True, help="Path to the target video file.")
    parser.add_argument("--image", type=str, required=True, help="Path to the source image file with the face to swap in.")

    args = parser.parse_args()

    # Basic validation
    if not os.path.exists(args.video):
        print(f"Error: Video file not found at '{args.video}'")
        sys.exit(1)
    if not os.path.exists(args.image):
        print(f"Error: Image file not found at '{args.image}'")
        sys.exit(1)

    logger.info("Starting application...")
    main(video_path=args.video, image_path=args.image)
    logger.info("Application finished.")
