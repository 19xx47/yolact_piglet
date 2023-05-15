import streamlit as st
import subprocess
import threading

def run_command(command, window_name, bbox_count):
    process = subprocess.Popen(command.split(), stdout=subprocess.PIPE)
    while True:
        output = process.stdout.readline()
        if output == b'' and process.poll() is not None:
            break
        if output:
            print(output.strip())

        bbox_count[window_name] += 1  # increment the bbox count for this thread

def main():
    st.title("Video Processing App")

    video1 = "20230411_20340101034709_20340101042649_115217.mp4"
    command1 = f"python eval.py --trained_model=./weights/piglet_detection_cam1_6857_480000.pth --config=yolact_im700_pig_config --score_threshold=0.15 --top_k=15 --video_multiframe=4 --video={video1} --display_fps"

    # Define the window names for each thread
    window1 = "Thread 1"
    
    # Create a shared bbox_count variable
    bbox_count = {window1: 0}

    # Start the subprocesses as threads
    thread1 = threading.Thread(target=run_command, args=(command1, window1, bbox_count))
    thread1.start()

    # Wait for the thread to finish
    thread1.join()

    # Print the bbox count
    st.write(f"Bbox count for {window1}: {bbox_count[window1]}")

if __name__ == "__main__":
    main()
