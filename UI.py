import streamlit as st
import cv2
import numpy as np
from insightface.app import FaceAnalysis
from insightface.model_zoo import get_model
from pathlib import Path
import os
import shutil
import time
from sklearn.cluster import DBSCAN
from insightface.utils import face_align
import ffmpeg
import tempfile
from io import BytesIO

# --- Configuration and Initialization ---

# Define the colors for the theme
PRIMARY_COLOR = '#E21F27'  # Primary Red/Orange
SECONDARY_COLOR = '#FF2D55'
GRADIENT_START = '#E21F27'
GRADIENT_END = '#FF9500'

# Global Model Pointers (Models are loaded once at the start)
if 'app' not in st.session_state:
    st.session_state.app = None
if 'swapper' not in st.session_state:
    st.session_state.swapper = None
if 'models_loaded' not in st.session_state:
    st.session_state.models_loaded = False

# Global State Management
if 'state' not in st.session_state:
    st.session_state.state = {
        'current_step': 1,
        'temp_video_path': None,
        'temp_image_path': None,
        'temp_audio_path': None,
        'clusters': {},
        'cluster_reps': [],
        'source_face': None,
        'video_metadata': {},
        'target_indices': [],
        'final_video_path': None,
    }

# Use a specific temp directory that we can manage
TEMP_DIR = Path(tempfile.gettempdir()) / "streamlit_faceswap_temp"
if not TEMP_DIR.exists():
    TEMP_DIR.mkdir(exist_ok=True)


# --- Utility Functions (Adapted for Streamlit) ---

@st.cache_resource
def load_models():
    """Loads InsightFace models and caches them."""
    try:
        st.info("Loading AI Models... This runs only once.")
        
        # Use a temporary directory for InsightFace models
        model_root = TEMP_DIR / "models"
        model_root.mkdir(exist_ok=True)
        
        app_instance = FaceAnalysis(name='buffalo_l', root=str(model_root))
        app_instance.prepare(ctx_id=0, det_size=(640, 640)) 
        
        st.info("Face Analysis model loaded. Loading Face Swapper model...")
        # NOTE: You must ensure 'weights/inswapper_128.onnx' is accessible
        # If the file is not found, an error will be raised here.
        swapper_instance = get_model('weights/inswapper_128.onnx', download=False, download_zip=False)
        
        st.session_state.app = app_instance
        st.session_state.swapper = swapper_instance
        st.session_state.models_loaded = True
        st.success("All models loaded successfully!")
        return True
    except Exception as e:
        st.error(f"Error loading models: {e}. Please ensure 'weights/inswapper_128.onnx' exists and dependencies are installed.")
        return False

def save_uploaded_file(uploaded_file, target_path):
    """Saves a streamlit uploaded file object to a temporary path."""
    with open(target_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return str(target_path)

def extract_audio(video_path, output_path):
    """Extracts audio from video using ffmpeg."""
    try:
        output_filename = Path(video_path).stem + "_audio.mp3"
        temp_audio_path = output_path / output_filename
        
        # Suppress command output unless an error occurs
        ffmpeg.input(video_path).output(str(temp_audio_path), acodec='mp3', loglevel="error").run(overwrite_output=True)
        return str(temp_audio_path)
    except ffmpeg.Error as e:
        # st.error(f"Error during audio extraction: {e.stderr.decode()}") # Suppress for cleaner output
        return None

def combine_video_audio(video_path, audio_path, output_path):
    """Combines a video stream with an audio stream using ffmpeg."""
    try:
        input_video = ffmpeg.input(video_path)
        input_audio = ffmpeg.input(audio_path)
        
        ffmpeg.concat(input_video, input_audio, v=1, a=1).output(
            str(output_path), 
            vcodec='libx264',
            acodec='aac',
            pix_fmt='yuv420p',
            loglevel="error"
        ).run(overwrite_output=True)
        return True
    except ffmpeg.Error as e:
        # st.error(f"Error during video/audio combination: {e.stderr.decode()}") # Suppress for cleaner output
        return False

# --- Core Processing Logic (Omitted for brevity, assumed functional) ---
# NOTE: The analyze_video_faces and process_face_swap functions remain exactly the same 
# as in the previous response, as their logic is correct.
# They are included in the downloadable file but commented out here to focus on the UI changes.

# def analyze_video_faces():
#     # ... (Functional code from previous response)
#     pass

# def process_face_swap(target_indices):
#     # ... (Functional code from previous response)
#     pass

# --- Streamlit UI Components (Sidebar) ---

def render_sidebar():
    # Updated App Name
    st.sidebar.markdown(f'<h1 style="color:{PRIMARY_COLOR};">Litzchill\'s Face-Morphing 🎭</h1>', unsafe_allow_html=True)
    st.sidebar.markdown("---")

    steps = {
        1: "Upload Video & Source Image",
        2: "Detected Faces & Selection",
        3: "Swapping Faces (Processing)",
        4: "Final Output Video",
    }
    
    current_step = st.session_state.state['current_step']

    for step, label in steps.items():
        if step == current_step:
            # Highlighted Style (Active Step)
            style = f"""
                background: linear-gradient(to right, {GRADIENT_START}, {GRADIENT_END});
                color: white;
                padding: 12px;
                border-radius: 10px;
                font-weight: bold;
                font-size: 1.15em;
                margin: 8px 0;
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.2);
            """
        else:
            # New Attractive Gray-Out Style (Inactive Step)
            style = f"""
                background-color: #f0f0f0; 
                color: #555;
                padding: 12px;
                border-radius: 10px;
                font-weight: 500;
                font-size: 1.15em;
                margin: 8px 0;
                border: 1px solid #ddd;
            """
        
        st.sidebar.markdown(f'<div style="{style}">{step}. {label}</div>', unsafe_allow_html=True)
        
    st.sidebar.markdown("---")
    
# --- Streamlit UI Components (Main Content - Omitted for brevity, logic unchanged) ---

# def render_step_1():
#     # ... (Functional code from previous response)
#     pass

# def render_step_2():
#     # ... (Functional code from previous response)
#     pass

# def render_step_3():
#     # ... (Functional code from previous response)
#     pass

# def render_step_4():
#     # ... (Functional code from previous response)
#     pass


# --- Main App Execution ---

def main():
    # Inject Custom CSS for layout and aesthetics
    st.markdown(
        f"""
        <style>
            /* Streamlit specific container adjustments for full height */
            .stApp {{
                max-width: none !important;
                padding: 0 !important;
            }}
            .block-container {{
                padding-top: 1rem;
                padding-bottom: 1rem;
                padding-left: 2rem;
                padding-right: 2rem;
            }}
            /* Custom spinner for Step 3 */
            .loader {{
                border: 16px solid #f3f3f3; 
                border-top: 16px solid {PRIMARY_COLOR}; 
                border-radius: 50%; 
                width: 120px; 
                height: 120px; 
                animation: spin 2s linear infinite; 
                margin: 20px auto;
            }}
            @keyframes spin {{ 
                0% {{ transform: rotate(0deg); }} 
                100% {{ transform: rotate(360deg); }} 
            }}
        </style>
        """,
        unsafe_allow_html=True
    )
    
    # 0. Load Models
    if not st.session_state.models_loaded:
        load_models()
        if not st.session_state.models_loaded:
             st.stop()
    
    # 1. Render Sidebar (20% of the screen)
    render_sidebar()
    
    # 2. Render Main Content based on step
    
    if st.session_state.state['current_step'] == 1:
        render_step_1()
    elif st.session_state.state['current_step'] == 2:
        render_step_2()
    elif st.session_state.state['current_step'] == 3:
        render_step_3()
    elif st.session_state.state['current_step'] == 4:
        render_step_4()

# Re-including core functions here to ensure the script is complete and runnable.

# --- Core Processing Logic ---
def analyze_video_faces():
    video_path = st.session_state.state['temp_video_path']
    app = st.session_state.app
    
    cap = cv2.VideoCapture(video_path)
    st.session_state.state['video_metadata']['fps'] = int(cap.get(cv2.CAP_PROP_FPS))
    st.session_state.state['video_metadata']['width'] = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    st.session_state.state['video_metadata']['height'] = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    all_faces = []
    all_embeddings = []
    frame_count = 0
    sampling_interval = 10
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    st.session_state.state['video_metadata']['total_frames'] = total_frames

    progress_bar = st.progress(0, text="Scanning video for faces (sampling every 10th frame)...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % sampling_interval == 0:
            detected_faces = app.get(frame)
            for face in detected_faces:
                face['frame_idx'] = frame_count
                all_faces.append(face)
                all_embeddings.append(face['embedding'])

        frame_count += 1
        progress_bar.progress(frame_count / total_frames, text=f"Scanning frame {frame_count}/{total_frames}...")
        
    cap.release()
    progress_bar.empty()

    if len(all_embeddings) == 0:
        raise ValueError("No faces detected in the video.")

    all_embeddings = np.array(all_embeddings)
    if all_embeddings.ndim == 1:
        all_embeddings = all_embeddings.reshape(1, -1)

    # Clustering
    clustering = DBSCAN(eps=0.6, min_samples=3, metric="cosine").fit(all_embeddings)
    labels = clustering.labels_

    clusters = {}
    for idx, label in enumerate(labels):
        if label == -1:
            continue
        if label not in clusters:
            clusters[label] = []
        clusters[label].append(all_faces[idx])

    if not clusters:
        raise ValueError("No valid face clusters (unique persons) found.")

    cluster_reps = []
    for label, faces in clusters.items():
        best_face = max(faces, key=lambda f: (f['bbox'][2] - f['bbox'][0]) * (f['bbox'][3] - f['bbox'][1]))
        cluster_reps.append((label, best_face))

    st.session_state.state['clusters'] = clusters
    st.session_state.state['cluster_reps'] = cluster_reps
    
def process_face_swap(target_indices):
    video_path = st.session_state.state['temp_video_path']
    source_face = st.session_state.state['source_face']
    app = st.session_state.app
    swapper = st.session_state.swapper
    metadata = st.session_state.state['video_metadata']

    cap = cv2.VideoCapture(video_path)
    fps = metadata['fps']
    width, height = metadata['width'], metadata['height']
    total_frames = metadata['total_frames']
    
    temp_video_output = TEMP_DIR / "swapped_temp_no_audio.mp4"
    final_video_output = TEMP_DIR / "final_swapped_output.mp4"

    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
    out = cv2.VideoWriter(str(temp_video_output), fourcc, fps, (width, height))
    
    target_embeddings = []
    if -1 not in target_indices:
        for idx in target_indices:
            if idx in st.session_state.state['clusters']:
                target_embeddings.extend([f['embedding'] for f in st.session_state.state['clusters'][idx]])
    
    frame_number = 0
    
    with st.status("Swapping Faces... (This is the slowest step, especially on CPU) 🐢", expanded=True) as status:
        status.update(label=f"Initializing video stream. Total frames: {total_frames}")
        
        progress_bar = st.progress(0)
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            detected_faces = app.get(frame)

            if detected_faces:
                for face in detected_faces:
                    perform_swap = False
                    
                    if -1 in target_indices:
                        perform_swap = True
                    else:
                        if target_embeddings:
                            new_embedding = face['embedding']
                            sims = [
                                np.dot(new_embedding, t) / 
                                (np.linalg.norm(new_embedding) * np.linalg.norm(t))
                                for t in target_embeddings
                            ]
                            if sims and max(sims) > 0.65:
                                perform_swap = True

                    if perform_swap:
                        frame = st.session_state.swapper.get(frame, face, source_face, paste_back=True)

            out.write(frame)
            frame_number += 1
            
            progress_percent = frame_number / total_frames
            progress_bar.progress(progress_percent)
            status.update(label=f"Processing frame {frame_number}/{total_frames} ({progress_percent:.1%})", state="running")

        cap.release()
        out.release()
        
        status.update(label="Combining video and audio with FFmpeg...", state="running")
        temp_audio_path = st.session_state.state['temp_audio_path']
        success = combine_video_audio(str(temp_video_output), temp_audio_path, str(final_video_output))

    if success:
        st.session_state.state['final_video_path'] = str(final_video_output)
        st.session_state.state['current_step'] = 4
    else:
        st.error("Failed to combine video and audio.")
        st.session_state.state['current_step'] = 2 
        
    st.rerun()

# --- Re-including Render Step Functions ---
def render_step_1():
    st.title("Step 1: Upload Video & Source Image")

    with st.form("upload_form"):
        video_file = st.file_uploader("Upload Input Video (.mp4, .mov)", type=['mp4', 'mov'])
        image_file = st.file_uploader("Upload Source Image (.jpg, .png)", type=['jpg', 'jpeg', 'png'])
        
        submitted = st.form_submit_button("Proceed to Detect Faces", type="primary")

    if submitted:
        if not video_file or not image_file:
            st.error("Please upload both a video and a source image.")
            return

        temp_video_path = save_uploaded_file(video_file, TEMP_DIR / video_file.name)
        temp_image_path = save_uploaded_file(image_file, TEMP_DIR / image_file.name)
        
        st.session_state.state['temp_video_path'] = temp_video_path
        st.session_state.state['temp_image_path'] = temp_image_path
        
        try:
            source_img = cv2.imread(temp_image_path)
            if source_img is None:
                raise ValueError("Could not read source image.")
            source_faces = st.session_state.app.get(source_img)
            if len(source_faces) == 0:
                raise ValueError("No face found in the source image.")
            st.session_state.state['source_face'] = source_faces[0]
            
            temp_audio_path = extract_audio(temp_video_path, TEMP_DIR)
            if not temp_audio_path:
                st.warning("Audio extraction failed. The final video will be silent.")
            st.session_state.state['temp_audio_path'] = temp_audio_path
            
            analyze_video_faces()
            
            st.session_state.state['current_step'] = 2
            st.rerun()
            
        except Exception as e:
            st.error(f"Processing Error: {e}")

def render_step_2():
    st.title("Step 2: Detected Faces & Selection")
    
    cols = st.columns([1, 3])
    with cols[0]:
        st.subheader("Source Face")
        source_img = cv2.imread(st.session_state.state['temp_image_path'])
        st.image(cv2.cvtColor(source_img, cv2.COLOR_BGR2RGB), caption="Source Face", use_column_width=True)

    with cols[1]:
        st.subheader("Detected Unique Persons in Video")
        cluster_reps = st.session_state.state['cluster_reps']
        
        if cluster_reps:
            rep_images_bytes = []
            captions = []
            video_path = st.session_state.state['temp_video_path']
            
            for label, face in cluster_reps:
                frame_idx = face['frame_idx']
                cap = cv2.VideoCapture(video_path)
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                cap.release()

                if ret:
                    aligned_face = face_align.norm_crop(frame, landmark=face['kps'])
                    is_success, buffer = cv2.imencode(".png", aligned_face)
                    if is_success:
                        rep_images_bytes.append(buffer.tobytes())
                        captions.append(f"Index {label}")
            
            st.image(rep_images_bytes, caption=captions, width=150)
            
            max_idx = max(st.session_state.state['clusters'].keys()) if st.session_state.state['clusters'] else -1
            
            index_input = st.text_input(
                f"Enter the index(es) of the person to swap (0 to {max_idx}, or -1 for all):",
                placeholder="e.g., 0, 1, 3 or -1"
            )

            if st.button("Morph Face", type="primary"):
                try:
                    raw_indices = [i.strip() for i in index_input.split(',')]
                    if not index_input or not raw_indices:
                        raise ValueError("Input cannot be empty. Please enter an index or -1.")
                        
                    target_indices = []
                    valid_indices = list(range(max_idx + 1))
                    
                    for i in raw_indices:
                        idx = int(i)
                        if idx != -1 and idx not in valid_indices:
                            raise ValueError(f"Invalid index: {idx}. Must be -1 or between 0 and {max_idx}.")
                        target_indices.append(idx)
                    
                    st.session_state.state['target_indices'] = target_indices
                    st.session_state.state['current_step'] = 3
                    st.rerun() 
                    
                except ValueError as e:
                    st.error(f"Input Error: {e}")

def render_step_3():
    st.title("Step 3: Swapping Faces (Processing)")
    
    st.markdown(
        f"""
        <div style="text-align: center;">
            <div class="loader"></div>
            <p style="font-size: 1.2em; color: {PRIMARY_COLOR};">
                **Processing... This is the video processing stage.**
                <br>If this takes too long, try a shorter, lower-resolution video.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    process_face_swap(st.session_state.state['target_indices'])

def render_step_4():
    st.title("Step 4: Final Output Video")
    
    final_video_path = st.session_state.state['final_video_path']
    if final_video_path and Path(final_video_path).exists():
        st.success("✅ Face swap complete! Your video is ready.")
        st.video(final_video_path)
        
        with open(final_video_path, "rb") as file:
            st.download_button(
                label="Download Final Video",
                data=file,
                file_name=Path(final_video_path).name,
                mime="video/mp4",
                type="primary"
            )
        
        st.markdown("---")
        if st.button("Start New Swap", type="secondary"):
             try:
                 shutil.rmtree(TEMP_DIR, ignore_errors=True)
             except Exception:
                 pass
             
             st.session_state.state = {
                'current_step': 1, 'temp_video_path': None, 'temp_image_path': None, 
                'temp_audio_path': None, 'clusters': {}, 'cluster_reps': [], 
                'source_face': None, 'video_metadata': {}, 'target_indices': [], 
                'final_video_path': None,
             }
             st.rerun()

if __name__ == "__main__":
    main()