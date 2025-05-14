import fitz  # PyMuPDF
import numpy as np
from PIL import Image
import json
import re
import io
import requests
import base64
from io import BytesIO
from transformers import AutoProcessor
from vllm import LLM, SamplingParams

def download_file(url, file_path):
    response = requests.get(url)
    with open(file_path, 'wb') as file:
        file.write(response.content)

url = "https://www.saffm.hq.af.mil/Portals/84/documents/FY25/FY25%20Air%20Force%20Working%20Capital%20Fund.pdf?ver=sHG_i4Lg0IGZBCHxgPY01g%3d%3d"
file_path = "file.pdf"
download_file(url, file_path)

# Load only page 30 from the PDF using PyMuPDF
def convert_page_to_image(pdf_path, page_number):
    doc = fitz.open(pdf_path)
    page = doc.load_page(page_number - 1)  # PyMuPDF uses 0-based indexing
    pix = page.get_pixmap(matrix=fitz.Matrix(150/72, 150/72))  # Lower resolution
    img_data = pix.tobytes("png")
    return Image.open(io.BytesIO(img_data))

# Function to convert PIL Image to base64 for vLLM
def image_to_base64(image):
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

# Get page 30 as an image
page_image = convert_page_to_image(file_path, 30)

MODEL_PATH = "Qwen/Qwen2.5-VL-7B-Instruct"
llm = LLM(
    model=MODEL_PATH,
    dtype="float16",  # Use float16 for better performance
    gpu_memory_utilization=0.8,  # Adjust based on your GPU
    limit_mm_per_prompt={"image": 10},
)

sampling_params = SamplingParams(
    temperature=0.1,
    top_p=0.001,
    repetition_penalty=1.05,
    max_tokens=512,  # Increased for better extraction
)

class Conductor:
    def __init__(self, llm, processor, images):
        self.llm = llm
        self.processor = processor
        self.images = images
    
    def process_image(self, image):
        # Convert the image to base64
        image_b64 = image_to_base64(image)
        
        # Create the messages with our specific extraction task
        messages = [
            {"role": "system", "content": "You are an expert at extracting text from images. Format your response in json."},
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image_url": {"url": f"data:image/png;base64,{image_b64}"},
                    },
                    {
                        "type": "text", 
                        "text": "Extract any numerical values from the page. If the page has a reference to its full value, as in 'Dollars in Millions' but the value is written as 3.15, extract it as 3150000. Just create a single list of all the values."
                    },
                ],
            },
        ]
        
        # Apply the chat template to get the prompt
        prompt = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        
        # Process the multi-modal data
        mm_data = {"image": [image_b64]}
        
        # Create the input for vLLM
        llm_inputs = {
            "prompt": prompt,
            "multi_modal_data": mm_data,
        }
        
        return llm_inputs
    
    def extract_values(self):
        results = {}
        
        for i, image in enumerate(self.images):
            # Process the image to get vLLM inputs
            llm_inputs = self.process_image(image)
            
            # Generate the response using vLLM
            outputs = self.llm.generate([llm_inputs], sampling_params=sampling_params)
            output = outputs[0].outputs[0].text
            
            # Extract values from the response
            try:
                # Try to parse as JSON
                json_output = json.loads(output)
                values = json_output['values'] if 'values' in json_output else json_output
            except json.JSONDecodeError:
                # Fallback to regex if not valid JSON
                values = re.findall(r'\d+(?:\.\d+)?', output)
            
            # Convert values to numbers
            results[i+1] = [self.convert_to_number(val) for val in values]
        
        return results
    
    def convert_to_number(self, val):
        if not isinstance(val, str):
            return float(val) if val is not None else None
        try:
            return float(val.replace(',', ''))
        except (ValueError, TypeError):
            return None
    
    def get_values_from_page(self):
        results = self.extract_values()
        return results

if __name__ == "__main__":
    # Initialize the processor
    processor = AutoProcessor.from_pretrained(MODEL_PATH)
    
    # Create a conductor instance
    conductor = Conductor(llm, processor, [page_image])
    
    # Extract values from page 30
    page_values = conductor.get_values_from_page()
    print(f"Values extracted from page 30: {page_values}")
    
    # Print the raw values in a more readable format
    if page_values and 1 in page_values:
        print("\nExtracted numerical values:")
        for i, value in enumerate(page_values[1]):
            print(f"{i+1}. {value}")
