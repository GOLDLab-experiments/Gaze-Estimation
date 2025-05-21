from transformers import AutoProcessor, PaliGemmaForConditionalGeneration, AutoModelForCausalLM, AutoTokenizer
from PIL import Image

print("Loading models...")

PALIGEMMA = 'google/paligemma-3b-ft-textcaps-448'
LLM_MODEL = 'meta-llama/Llama-3.2-1B-Instruct'
# LLM_MODEL = "facebook/opt-125m"

class VLM:
    def __init__(self, model_id=PALIGEMMA, device='cpu'):
        self.processor = AutoProcessor.from_pretrained(model_id, use_fast=True)
        self.model = PaliGemmaForConditionalGeneration.from_pretrained(model_id)
        self.model.to(device)
        self.device = device
        print("VLM model loaded and moved to", device)

    def analyze_image(self, image_path):
        prompt = "<image> Describe the person in the image. Consider their facial expression, mood, any objects they are holding, how they are dressed, and the environment. If there are multiple people, refer to the closest one."
        image = Image.open(image_path).convert("RGB")
        print("Image loaded.")
        inputs = self.processor(text=prompt, images=image, return_tensors='pt')
        self.model.eval()
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        print("Inputs processed and moved to", self.device)
        generated_ids = self.model.generate(
            **inputs,
            max_new_tokens=150,
            num_beams=1,
            do_sample=False
        )
        description = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip().lower()
        print("Analysis Result:", description)
        return description

class LLM:
    def __init__(self, model_id=LLM_MODEL, device='cpu'):
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
    image_path = "test_image.jpg"
    vlm = VLM()
    llm = LLM()
    image_description = vlm.analyze_image(image_path)
    classification = llm.classify(image_description)
    # Use the LLM output as additional input to the LLM (demonstration)
    refined_classification = llm.classify(image_description, previous_output=classification)
    print(f"Refined Classification Result: {refined_classification}")