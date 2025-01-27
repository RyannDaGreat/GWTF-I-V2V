import gradio as gr
import subprocess
import shutil
import os

is_shared_ui = True if "fffiloni/Go-With-The-Flow" in os.environ['SPACE_ID'] else False

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

def process_video(video_path, prompt, num_steps, degradation_level):
    
    output_folder="noise_warp_output_folder"
    
    if os.path.exists(output_folder):
        # Delete the folder and its contents
        shutil.rmtree(output_folder)
    # Check if the file exists and delete it
    if os.path.exists("output.mp4"):
        os.remove("output.mp4")
    
    output_video="output.mp4"
    device="cuda"
    
    try:
        # Step 1: Warp the noise
        gr.Info("Step 1: Warp the noise...")
        warp_command = [
            "python", "make_warped_noise.py", video_path,
            "--output_folder", output_folder
        ]
        subprocess.run(warp_command, check=True)

        warped_vid_path = os.path.join(output_folder, "input.mp4")
        
        # Step 2: Run inference
        gr.Info("Step 2: Run inference...")
        inference_command = [
            "python", "cut_and_drag_inference.py", output_folder,
            "--prompt", prompt,
            "--degradation", str(degradation_level),
            "--output_mp4_path", output_video,
            "--device", device,
            "--num_inference_steps", str(num_steps)
        ]
        subprocess.run(inference_command, check=True)
        
        # Return the path to the output video
        gr.Success("Done!")
        return output_video
    except subprocess.CalledProcessError as e:
        
        raise gr.Error(f"An error occurred: {str(e)}")

css="""
div#follow-div{
    text-decoration: none !important;
    display: flex;
    column-gap: 5px;
    font-size: 0.8em;
}
"""

with gr.Blocks(css=css) as demo:
    with gr.Column():
        gr.Markdown("# Go-With-The-Flow • Cut and Drag")
        gr.HTML("""
        <div style="display:flex;column-gap:4px;">
            <a href="https://github.com/Eyeline-Research/Go-with-the-Flow">
                <img src='https://img.shields.io/badge/GitHub-Repo-blue'>
            </a> 
            <a href="https://arxiv.org/abs/2501.08331">
                <img src='https://img.shields.io/badge/ArXiv-Paper-red'>
            </a>
            <a href="https://eyeline-research.github.io/Go-with-the-Flow/">
                <img src='https://img.shields.io/badge/Project-Page-green'>
            </a>
            <a href="https://huggingface.co/spaces/fffiloni/Go-With-The-Flow?duplicate=true">
                <img src="https://huggingface.co/datasets/huggingface/badges/resolve/main/duplicate-this-space-sm.svg" alt="Duplicate this Space">
            </a>
        </div>
        """)
        with gr.Row():
            with gr.Column():
                input_video = gr.Video(label="Input Video")
                prompt = gr.Textbox(label="Prompt")
                with gr.Row():
                    if is_shared_ui:
                        num_steps = gr.Slider(label="Inference Steps", minimum=1, maximum=30, value=5, step=1, interactive=False)
                        degradation = gr.Slider(label="Noise Degradation", minimum=0, maximum=1, value=0.5, step=0.1, interactive=False)
                    else:
                        num_steps = gr.Slider(label="Inference Steps", minimum=1, maximum=30, value=20, step=1, interactive=True)
                        degradation = gr.Slider(label="Noise Degradation", minimum=0, maximum=1, value=0.5, step=0.1, interactive=True)
                    
                submit_btn = gr.Button("Submit")
                gr.Examples(
                    examples = [
                        ["./examples/example_1.mp4", "yellow plastic duck is swimming and jumping in the water"],
                        ["./examples/example_2.mp4", "a car enters the frame and goes forward to the end of the street"]
                    ], 
                    inputs = [input_video, prompt]
                )
            with gr.Column():
                output_video = gr.Video(label="Result")
                
                gr.HTML("""
                <div id="follow-div">
                    <a href="https://huggingface.co/fffiloni">
                        <img src="https://huggingface.co/datasets/huggingface/badges/resolve/main/follow-me-on-HF-sm-dark.svg" alt="Follow me on HF">
                    </a>
                    <p>for space updates</p>
                """)

    submit_btn.click(
        fn = process_video,
        inputs = [input_video, prompt, num_steps, degradation],
        outputs = [output_video]
    )

demo.queue().launch(show_api=False)