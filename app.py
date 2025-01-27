import gradio as gr
import subprocess
import os

from huggingface_hub import snapshot_download

# Define the folder name
folder_name = "lora_models"

# Create the folder
os.makedirs(folder_name, exist_ok=True)

# Download models
snapshot_download(
    repo_id = "Eyeline-Research/Go-with-the-Flow",
    local_dir = folder_name
)

def process_video(video_path, prompt):
    
    output_folder="noise_warp_output_folder"
    output_video="output.mp4"
    device="cuda"
    num_steps=5
    
    try:
        # Step 1: Warp the noise
        warp_command = [
            "python", "make_warped_noise.py", video_path,
            "--output_folder", output_folder
        ]
        subprocess.run(warp_command, check=True)
        
        # Step 2: Run inference
        inference_command = [
            "python", "cut_and_drag_inference.py", output_folder,
            "--prompt", prompt,
            "--output_mp4_path", output_video,
            "--device", device,
            "--num_inference_steps", str(num_steps)
        ]
        subprocess.run(inference_command, check=True)
        
        # Return the path to the output video
        return output_video
    except subprocess.CalledProcessError as e:
        
        raise gr.Error(f"An error occurred: {str(e)}")

with gr.Blocks() as demo:
    with gr.Column():
        gr.Markdown("# Go-With-The-Flow")
        with gr.Row():
            with gr.Column():
                input_video = gr.Video(label="Input Video")
                prompt = gr.Textbox(label="Prompt")
                submit_btn = gr.Button("Submit")
            with gr.Column():
                output_video = gr.Video(label="Result")

    submit_btn.click(
        fn = process_video,
        inputs = [input_video, prompt],
        outputs = [output_video]
    )

demo.queue().launch(show_api=False)