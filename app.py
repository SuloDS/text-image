import streamlit as st
from diffusers import DiffusionPipeline
import torch

if not torch.cuda.is_available():
    st.warning("CUDA is not available. You may experience slower performance.")

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load the pipeline with proper handling of potential memory issues
try:
    pipe = DiffusionPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=torch.float16,
        use_safetensors=True,
        variant="fp16",
        revision="7f31e00",  # Specify a stable revision for consistency
        device=device,
    )
except RuntimeError as e:
    if "CUDA out of memory" in str(e):
        st.error("CUDA memory is insufficient. Consider downgrading the model revision or using a CPU device.")
    else:
        raise e

# Add a safety check for model revision compatibility
if pipe.revision != "7f31e00":
    st.warning("The model revision provided may not be compatible with the current pipeline. Please adjust the revision if necessary.")

# Create the Streamlit app
st.title("Image Generation App")

# Add an input field for the prompt
prompt_text = st.text_input("Enter your prompt:", value="An astronaut riding a green horse")

# Add options for image size
image_size = st.selectbox(
    "Image Size:",
    options=[128, 256, 512, 768],
    index=2,  # Default to 512
)

# Add a button to generate the image
if st.button("Generate Image"):
    # Generate image using the specified prompt and size
    with st.spinner("Generating image..."):
        try:
            images = pipe(prompt=prompt_text, size=image_size, num_inference_steps=50).images[0]
        except Exception as e:
            st.error(f"Error generating image: {str(e)}")
            continue

    # Display the generated image
    st.image(images)

# Display usage instructions and disclaimer
st.markdown(
    """**Instructions:**

    * Enter a text prompt describing the image you want to generate.
    * Select the desired image size.
    * Click the "Generate Image" button.

    **Disclaimer:**

    * This app is for demonstration purposes only.
    * It may take some time to generate the image, depending on the complexity of the prompt and your device's resources.
    * The results may not always be perfect, and the model may generate unexpected or offensive content. Use it responsibly and at your own risk.
    """
)

