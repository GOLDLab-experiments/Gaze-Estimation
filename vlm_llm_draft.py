import torch
from PIL import Image
# from transformers import AutoProcessor, PaliGemmaForConditionalGeneration, AutoModelForCausalLM, AutoTokenizer
from transformers import AutoProcessor, AutoModelForVision2Seq, BitsAndBytesConfig
# from transformers.image_utils import load_image
from transformers import AutoProcessor, AutoModelForCausalLM, AutoTokenizer
from time import time

print("Loading models...")

# PALIGEMMA = 'google/paligemma-3b-ft-textcaps-448'
LLM_MODEL = 'meta-llama/Llama-3.2-1B-Instruct'
# LLM_MODEL = "facebook/opt-125m"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class VLM:
    def __init__(self, device=DEVICE):
        self.processor = AutoProcessor.from_pretrained("HuggingFaceTB/SmolVLM-500M-Instruct")
        self.model = AutoModelForVision2Seq.from_pretrained(
            "HuggingFaceTB/SmolVLM-500M-Instruct",
            torch_dtype=torch.bfloat16,
            # _attn_implementation="flash_attention_2" if device == "cuda" else "eager",
        ).to(device)
        print("VLM model loaded and moved to", device)

    def analyze_image(self, image_path):
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": "Describe the person in the image. Consider their facial expression, mood, any objects they are holding, how they are dressed, and the environment. If there are multiple people, refer to the closest one."}
                ]
            },
        ]
        # prompt = "<image> Describe the person in the image. Consider their facial expression, mood, any objects they are holding, how they are dressed, and the environment. If there are multiple people, refer to the closest one."
    
        image = Image.open(image_path).convert("RGB")
        print("Image loaded.")

        prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True)

        inputs = self.processor(text=prompt, images=[image], return_tensors='pt')
        inputs = inputs.to(DEVICE)

        # self.model.eval()
        # inputs = {k: v.to(self.device) for k, v in inputs.items()}
        # print("Inputs processed and moved to", self.device)
        # generated_ids = self.model.generate(
        #     **inputs,
        #     max_new_tokens=150,
        #     num_beams=1,
        #     do_sample=False
        # )
        # description = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip().lower()
        # print("Analysis Result:", description)

        print ("Starting to generate outputs...")
        # Generate outputs
        generated_ids = self.model.generate(**inputs, max_new_tokens=500)
        # quantization_config = BitsAndBytesConfig(load_in_8bit=True)
        generated_texts = self.processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
        #    quantization_config=quantization_config
        )
        description = generated_texts[0].strip()
        description = description.partition("Assistant:")[2].lstrip()

        print("This is the description.\n")
        print(description)

        print("\nEnd of desription")
        return description

class LLM:
    def __init__(self, model_id=LLM_MODEL, device=DEVICE):
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
        self.model = AutoModelForCausalLM.from_pretrained(model_id)
        self.model.to(device)
        self.device = device
        print(f"LLM ({model_id}) loaded and moved to {device}.")
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def classify(self, image_description, previous_output=None):
        classification_prompt = (
            "You are an AI model that classifies images based on their descriptions.\n"
            "Based on the camera image description provided, classify the image into one of the following categories:\n"
            "1. Benign, 2. Malicious, 3. Authorized\n"
        )
        if previous_output:
            classification_prompt += f"\nPrevious classification: {previous_output}\n"
        full_prompt = f"{classification_prompt}\nDescription: {image_description}\nClassification:"
        inputs = self.tokenizer(full_prompt, return_tensors='pt')
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        print("Inputs processed and moved to", self.device, "for LLM.")
        generated_ids = self.model.generate(
            **inputs,
            max_new_tokens=10,
            num_beams=1,
            do_sample=False,
            pad_token_id=self.tokenizer.pad_token_id
        )
        output = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
        print("LLM Output:", output)
        # Extract classification
        predicted_class = "uncertain"
        classification_start_index = output.lower().find("classification:")
        if classification_start_index != -1:
            start_of_class_text = output[classification_start_index + len("classification:"):].strip()
            if start_of_class_text:
                predicted_class = start_of_class_text.split()[0].lower()
        else:
            response_lower = output.lower()
            if "malicious" in response_lower:
                predicted_class = "malicious"
            elif "benign" in response_lower:
                predicted_class = "benign"
            elif "authorized" in response_lower:
                predicted_class = "authorized"
        if predicted_class not in ["malicious", "benign", "authorized", "uncertain"]:
            predicted_class = "uncertain (unrecognized output)"
        print(f"Classification Result: {predicted_class}")
        return predicted_class

if __name__ == "__main__":

    start = time()

    image_path = "test_image.jpg"
    vlm = VLM()
    llm = LLM()
    image_description = vlm.analyze_image(image_path)

    vlm_time = time()
    print(f"VLM analysis completed in {vlm_time - start:.2f} seconds.")

    classification = llm.classify(image_description)
    # Use the LLM output as additional input to the LLM (demonstration)
    # refined_classification = llm.classify(image_description, previous_output=classification)
    # print(f"Refined Classification Result: {refined_classification}")

    end = time()
    print(f"Total time taken: {end - start:.2f} seconds")
    print("Processing complete.")