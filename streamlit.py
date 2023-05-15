from ffmpeg_progress_yield import FfmpegProgress
import streamlit as st
import torch
from PIL import Image
import wget
import time
import subprocess
import getpass

# Import your eval function from eval.py
from eval import eval  # Assuming your eval function is named eval

# Define the model path
trained_model = "./weights/piglet_detection_cam1_6857_480000.pth"
config = "yolact_im700_pig_config"
score_threshold = 0.15
top_k = 15
video_multiframe = 4
video = "20230411_20340101034709_20340101042649_115217.mp4:output_video.mp4"
display_fps = True

# Define whether to enable URL download
cfg_enable_url_download = False
if cfg_enable_url_download:
    url = "https://archive.org/download/yoloTrained/yoloTrained.pt"
    trained_model = f"models/{url.split('/')[-1]}"

# Helper function to load the model from URL
@st.cache(allow_output_mutation=True)
def load_model():
    if url:
        model_file = wget.download(url, out="models/")
    else:
        model_file = trained_model
    return model_file

# Main function
def main():
    # -- Sidebar
    st.sidebar.title('⚙️Options')
    datasrc = st.sidebar.radio("Select input source.", ['From test set.', 'Upload your own data.'])
    option = st.sidebar.radio("Select input type.", ['Video'])
    
    if torch.cuda.is_available():
        deviceoption = st.sidebar.radio("Select compute Device.", ['cpu', 'cuda'], disabled=False, index=1)
    else:
        deviceoption = st.sidebar.radio("Select compute Device.", ['cpu', 'cuda'], disabled=True, index=0)
    # -- End of Sidebar

    st.header('📦Obstacle Detection')
    st.subheader('👈🏽 Select options left-handed menu bar.')
    st.sidebar.markdown("https://github.com/thepbordin/Obstacle-Detection-for-Blind-people-Deployment")
    
    if option == "Video":
        videoInput(deviceoption, datasrc)

# Function to handle video input
def videoInput(device, src):
    uploaded_video = st.file_uploader("Upload Video", type=['mp4', 'mpeg', 'mov'])
    if uploaded_video is not None:
        ts = time.time()
        video_path = f"uploaded_video_{ts}.mp4:output_video.mp4"
        video = "uploaded_video_{ts}.mp4:output_video.mp4"

        output_path = "output_video.mp4"

        with open(video_path, mode='wb') as f:
            f.write(uploaded_video.read())  # save video to disk

        st.video(video_path)
        st.write("Uploaded Video")

        # Load the model
        model_file = trained_model

        # Run the evaluation
        eval(model_file, config, score_threshold, top_k, video_multiframe, video, display_fps)
    
        sudo_password = getpass.getpass("Enter your sudo password: Wmai191247")

            # Construct the ffmpeg command
        ffmpeg_cmd = [
            "sudo",
            "-S",
            "ffmpeg",
            "-i",
            output_path,
            "-vcodec",
            "libx264",
            "output_video_new.mp4"
        ]

            # Run the command and provide the sudo password programmatically
        process = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
        process.communicate(input=sudo_password+ '\n')
    
        # subprocess.run(["sudo", "ffmpeg", "-i", output_path, "-vcodec", "libx264", f"output_video_new.mp4"])
        video_path_output = "output_video_new.mp4"
        st.video(video_path_output)
        st.write("Model Prediction")

if __name__ == '__main__':
    main()
