from dataclasses import dataclass

@dataclass
class FaceDetectionArtifact:
    detected_faces_path: str
    json_information: str
    extracted_audio_path: str

@dataclass
class FaceSwappingArtifact:
    final_output_video_path: str