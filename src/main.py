import argparse
from src.pipeline.pipeline import FaceSwapPipeline
from src.loggings.logger import logger

def main():
    parser = argparse.ArgumentParser(description="Run the Face Swap Pipeline.")
    parser.add_argument("--video_path", type=str, required=True, help="Path to the input video file.")
    parser.add_argument("--image_path", type=str, required=True, help="Path to the source image file for face swapping.")
    parser.add_argument("--cluster_id", type=int, default=None, help="Optional: Specific cluster ID to swap (otherwise, uses the most prominent).")

    args = parser.parse_args()

    pipeline = FaceSwapPipeline(video_path=args.video_path, image_path=args.image_path)
    final_video_path = pipeline.run(user_cluster_id=args.cluster_id)

    if final_video_path:
        logger.info(f"Pipeline executed successfully. Output: {final_video_path}")

if __name__ == "__main__":
    main()