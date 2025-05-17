from transformers import AutoProcessor, PaliGemmaForConditionalGeneration, AutoModelForCausalLM, AutoTokenizer
from PIL import Image
from huggingface_hub import login

from hf_token import get_token # This module is in .gitignore to not publish the token.
login(get_token())  # Log in to HuggingFace Hub using a private token

# Paligemma model for image captioning
# This model is a fine-tuned version of the PaliGemma model for image captioning tasks.
PALIGEMMA = 'google/paligemma-3b-ft-textcaps-448'

# Load the processor and model for image captioning
processor = AutoProcessor.from_pretrained(PALIGEMMA)
paligemma = PaliGemmaForConditionalGeneration.from_pretrained(PALIGEMMA)

paligemma.to('cpu')  # Move model to CPU
print("Model loaded and moved to CPU.")

# Prompt for the image captioning model
paligemma_prompt = """
You are a vision model for a dog-like robot. The robot is on a mission. 
As it preforms its task, it encounters a person. The robot needs to analyze the person
Analyze the person in the image. Consider their facial expression,mood,
any objects they are holding, their dressing, and the surrounding environment.
If there are multiple people, refer to the closest one."""

image_url = "test_image.jpg"  # Path to the image file
image = Image.open(image_url).convert("RGB")  # Load and convert image to RGB
print("Image loaded.")

# Prepare inputs for the image captioning model
inputs = processor(text=paligemma_prompt, images=image, return_tensors='pt')

# Move all input tensors to CPU
inputs = {k: v.to('cpu') for k, v in inputs.items()}
print("Inputs processed and moved to CPU.")

# Generate the image description using the captioning model
generated_ids = paligemma.generate(
    **inputs,
    max_new_tokens=150,  # Allow up to 150 tokens for a paragraph-length description
    num_beams=1,
    do_sample=False
)
image_description = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip().lower()

print("Analysis Result:", image_description)

# LLM model for classification
LLAMA = "meta-llama/Llama-3.2-1B-Instruct"
# Or for a smaller, faster option: llm_model_id = "facebook/opt-125m"

# Load tokenizer and model for the LLM
llama_tokenizer = AutoTokenizer.from_pretrained(LLAMA)
llama_model = AutoModelForCausalLM.from_pretrained(LLAMA)

llama_model.to('cpu')  # Move LLM to CPU
print(f"\nLLM ({LLAMA}) loaded and moved to CPU.")

# Prompt for the LLM to classify the image description
classification_prompt = """
You are an AI model that classifies images based on their descriptions.
Based on the camera image description provided, classify the image into one of the following categories:
1. Benign, 2. Malicious, 3. Authorized
"""

# Combine the classification prompt with the image description
# Adding a clear separator can help the LLM understand the different parts
full_llm_prompt = f"{classification_prompt}\n\nDescription: {image_description}\n\nClassification:"

# Ensure the tokenizer has a padding token
if llama_tokenizer.pad_token is None:
    llama_tokenizer.pad_token = llama_tokenizer.eos_token

# Tokenize the prompt for the LLM
inputs = llama_tokenizer(full_llm_prompt, return_tensors='pt')

# Move all input tensors to CPU
inputs = {k: v.to('cpu') for k, v in inputs.items()}
print("Inputs processed and moved to CPU for LLM.")

# Generate the classification output
# Set max_new_tokens to a small number since you expect a single word output
generated_ids = llama_model.generate(
    **inputs,
    max_new_tokens=10, # Expecting a short output like a single word
    num_beams=1,       # Simple greedy decoding
    do_sample=False,   # No sampling
    pad_token_id=llama_tokenizer.pad_token_id # Use padding token if defined
)

# Decode the output from the LLM
classification_output = llama_tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()

# Post-process the output to extract only the classification word
# This is a simple approach; you might need more robust parsing
# based on how the LLM responds. The LLM might repeat the prompt or add extra text.
# We'll try to find the text immediately after the "Classification:" part.

predicted_class = "uncertain" # Default if classification is not found

classification_start_index = classification_output.lower().find("classification:")
if classification_start_index != -1:
    # Find the first word after "Classification:"
    # Add length of "Classification:" and potentially some space characters
    start_of_class_text = classification_output[classification_start_index + len("classification:"):].strip()
    if start_of_class_text:
        predicted_class = start_of_class_text.split()[0].lower()
else:
    # If the LLM didn't follow the format, try to find one of the expected words
    response_lower = classification_output.lower()
    if "malicious" in response_lower:
        predicted_class = "malicious"
    elif "benign" in response_lower:
        predicted_class = "benign"
    elif "authorized" in response_lower:
        predicted_class = "authorized"

# Basic check if the predicted class is one of the expected values
if predicted_class not in ["malicious", "benign", "authorized", "uncertain"]:
    predicted_class = "uncertain (unrecognized output)"

print(f"Classification Result: {predicted_class}")