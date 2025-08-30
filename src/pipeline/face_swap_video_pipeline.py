import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from insightface.utils import face_align

from src.loggings.logger import logger
from src.components.face_detection import DetectFaces
from src.components.face_swapper import SwapFaces
from src.entity.artifact_entity import FaceDetectionArtifact

class FaceSwapPipeline:
    def __init__(self, video_path: str):
        self.video_path = video_path
        self.detection_artifact = None
        self.clusters = None
        self.face_detector = None

    def detect_faces(self):
        """Detects faces in the video and clusters them."""
        logger.info("Starting face detection and clustering...")
        self.face_detector = DetectFaces(video_path=self.video_path)
        self.detection_artifact, self.clusters = self.face_detector.video_preprocessing()

        if not self.clusters:
            logger.warning("No valid clusters found in the video.")
            return False
        
        logger.info(f"Found {len(self.clusters)} unique persons.")
        return True

    def display_and_select_faces(self):
        """Displays unique faces and prompts user for selection."""
        if not self.clusters:
            logger.warning("No clusters to display.")
            return None, None

        cluster_reps = []
        for label, faces in sorted(self.clusters.items()):
            best_face = max(
                faces,
                key=lambda f: (f['bbox'][2] - f['bbox'][0]) * (f['bbox'][3] - f['bbox'][1])
            )
            cluster_reps.append((label, best_face))

        num_faces = len(cluster_reps)
        if num_faces > 0:
            cols = min(5, num_faces)
            rows = (num_faces + cols - 1) // cols
            fig, axes = plt.subplots(rows, cols, figsize=(15, 3 * rows))

            if num_faces == 1:
                axes = np.array([[axes]])
            elif rows == 1:
                axes = axes.reshape(1, -1)

            for i in range(num_faces):
                row = i // cols
                col = i % cols
                label, face = cluster_reps[i]
                frame_idx = face['frame_idx']
                cap = cv2.VideoCapture(self.video_path)
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                cap.release()
                if ret:
                    aligned_face = face_align.norm_crop(frame, landmark=face['kps'])
                    axes[row, col].imshow(cv2.cvtColor(aligned_face, cv2.COLOR_BGR2RGB))
                axes[row, col].set_title(f"Index {label}", fontsize=12)
                axes[row, col].axis('off')

            for i in range(num_faces, rows * cols):
                row = i // cols
                col = i % cols
                axes[row, col].axis('off')

            plt.tight_layout()
            plt.show()

            print("--- Unique Persons Found ---")
            for label, face in cluster_reps:
                bbox = face['bbox'].astype(int)
                print(f"Index {label}: Representative face at [x1:{bbox[0]}, y1:{bbox[1]}, x2:{bbox[2]}, y2:{bbox[3]}]")
            print("Index -1: Swap ALL persons")

        source_face_path = input("Enter the path to your source image: ").strip().strip('"')

        if not Path(source_face_path).exists():
            logger.error(f"Source image '{source_face_path}' not found. Exiting.")
            return None, None

        source_img = cv2.imread(source_face_path)
        
        if self.face_detector is None:
            logger.error("Face detector not initialized. Run detect_faces first.")
            return None, None
            
        source_faces = self.face_detector.face_detection_model.get(source_img)
        if len(source_faces) == 0:
            print("No face found in the source image. Exiting.")
            return None, None

        plt.figure(figsize=(5, 5))
        plt.imshow(cv2.cvtColor(source_img, cv2.COLOR_BGR2RGB))
        plt.title("Source Face", fontsize=14)
        plt.axis('off')
        plt.show()

        try:
            index = int(input(f"Enter the index of the person to swap (0 to {len(self.clusters)-1}, or -1 for all): "))
            if index < -1 or (index >= len(self.clusters) and index != -1):
                print("Invalid index. Exiting.")
                return None, None
        except ValueError:
            logger.error("Invalid index input. Must be an integer.")
            return None, None
            
        return source_face_path, index

    def swap_faces(self, source_face_path: str, index: int):
        """Swaps faces in the video based on user selection."""
        if self.detection_artifact is None or self.clusters is None:
            logger.error("Face detection must be run before swapping.")
            return

        logger.info(f"Preparing to swap faces for index {index}...")

        if index == -1:
            swap_clusters = [face for cluster_faces in self.clusters.values() for face in cluster_faces]
        else:
            swap_clusters = self.clusters.get(index, [])

        if not swap_clusters:
            logger.warning(f"No faces found for index {index}. Nothing to swap.")
            return

        FaceDetectionArtifact.extracted_audio_path = self.detection_artifact.extracted_audio_path

        sf = SwapFaces(index=index, video_path=self.video_path, source_face_path=source_face_path, clusters=swap_clusters)
        swapping_artifact = sf.video_preprocessing()
        
        if swapping_artifact and hasattr(swapping_artifact, 'output_video_path'):
            logger.info(f"Face swapping complete. Output video at: {swapping_artifact.output_video_path}")
        else:
            logger.error("Face swapping failed.")

    def run_full_pipeline(self):
        """Runs the complete face detection and swapping pipeline."""
        if self.detect_faces():
            source_face_path, index = self.display_and_select_faces()
            if source_face_path is not None and index is not None:
                self.swap_faces(source_face_path, index)

# if __name__ == "__main__":
#     # This main function is for testing the pipeline class directly if needed.
#     # The main application entry point will be in src/main.py
#     video_path = "path/to/your/video.mp4" # Example path
#     pipeline = FaceSwapPipeline(video_path=video_path)
#     pipeline.run_full_pipeline()
