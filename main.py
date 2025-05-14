import base64
import json
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch
import fitz  # PyMuPDF
import io
import time
import re
import os

class PDFAnalyzer:
    def __init__(self, pdf_path, output_dir="pdf_pages", batch_size=5):
        """
        Initialize the PDFAnalyzer with the path to the PDF and configuration.
        
        Args:
            pdf_path (str): Path to the PDF file
            output_dir (str): Directory to store converted page images
            batch_size (int): Number of pages to process in each batch
        """
        self.pdf_path = pdf_path
        self.output_dir = output_dir
        self.batch_size = batch_size
        self.model = None
        self.processor = None
        self.page_paths = []
        
    def _initialize_model(self):
        """Initialize the Qwen model and processor if not already initialized."""
        if self.model is None:
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                "Qwen/Qwen2.5-VL-7B-Instruct",
                torch_dtype=torch.bfloat16,
                attn_implementation="flash_attention_2",
                device_map="auto",
            )
            
            min_pixels = 384 * 28 * 28
            max_pixels = 768 * 28 * 28
            self.processor = AutoProcessor.from_pretrained(
                "Qwen/Qwen2.5-VL-7B-Instruct",
                min_pixels=min_pixels,
                max_pixels=max_pixels
            )

    def convert_pdf_pages_to_images(self):
        """Convert PDF pages to high-quality images."""
        os.makedirs(self.output_dir, exist_ok=True)
        
        doc = fitz.open(self.pdf_path)
        self.page_paths = []
        
        for i, page in enumerate(doc):
            matrix = fitz.Matrix(3, 3)  # 3x zoom for better resolution
            pix = page.get_pixmap(
                matrix=matrix,
                alpha=False,
            )
            
            output_path = os.path.join(self.output_dir, f"page_{i+1}.png")
            pix.save(output_path)
            self.page_paths.append(output_path)
        
        doc.close()
        return self.page_paths

    def process_pdf_pages(self):
        """Process the PDF pages and extract numerical values."""
        self._initialize_model()
        start_time = time.time()
        
        all_numbers = []
        total_pages = len(self.page_paths)
        total_processing_time = 0
        
        for i in range(0, total_pages, self.batch_size):
            batch_start_time = time.time()
            batch_pages = self.page_paths[i:i + self.batch_size]
            print(f"\nProcessing pages {i+1} to {min(i+self.batch_size, total_pages)} of {total_pages}...")
            
            messages = self._prepare_batch_messages(batch_pages)
            batch_numbers = self._process_batch(messages, i)
            
            all_numbers.extend(batch_numbers)
            
            batch_end_time = time.time()
            batch_time = batch_end_time - batch_start_time
            total_processing_time += batch_time
            
            self._print_batch_stats(batch_numbers, batch_time, len(batch_pages), len(all_numbers))
        
        total_time = time.time() - start_time
        return all_numbers, total_time, total_processing_time

    def _prepare_batch_messages(self, batch_pages):
        """Prepare messages for batch processing."""
        messages = []
        for page_path in batch_pages:
            message = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": f"file://{page_path}"},
                        {"type": "text", "text": """
                        Extract ONLY the largest numerical value on this financial page.
                        
                        Check for scale indicators:
                        - "Dollars in Millions" → multiply by 1,000,000
                        - "Dollars in Thousands" → multiply by 1,000
                        - "($ in millions)" → multiply by 1,000,000
                        
                        Return ONLY the final value after applying any multiplier.
                        Do not include any decimals or commas in the final value.
                        Do not explain the value, just return the value.
                        If no value is found, return 0.
                        """},
                    ],
                }
            ]
            messages.append(message)
        return messages

    def _process_batch(self, messages, batch_start_idx):
        """Process a batch of pages and extract numbers."""
        texts = [
            self.processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
            for msg in messages
        ]

        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=texts,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self.model.device)

        generated_ids = self.model.generate(
            **inputs,
            max_new_tokens=1024,
            do_sample=False,
        )
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

        batch_numbers = []
        for batch_idx, text in enumerate(output_text):
            try:
                text = text.strip().replace('$', '').replace(',', '')
                if '%' in text:
                    continue
                
                number = float(text)
                if number > 0:
                    page_number = batch_start_idx + batch_idx + 1
                    batch_numbers.append((page_number, number))
            except (ValueError, TypeError) as e:
                print(f"Warning: Could not parse number from response: {text}")
                print(f"Error details: {str(e)}")
                continue
                
        return batch_numbers

    def _print_batch_stats(self, batch_numbers, batch_time, batch_size, total_numbers):
        """Print statistics for the current batch."""
        if batch_numbers:
            max_batch_page, max_batch_value = max(batch_numbers, key=lambda x: x[1])
            print(f"Batch results:")
            print(f"  - Largest number in batch: {max_batch_value:,.0f} (on page {max_batch_page})")
            print(f"  - Total numbers found in batch: {len(batch_numbers)}")
        else:
            print("No numbers found in this batch")
        
        avg_time_per_page = batch_time / batch_size
        print(f"Found {total_numbers} numbers so far...")
        print(f"Batch processing time: {batch_time:.2f} seconds")
        print(f"Average time per page in this batch: {avg_time_per_page:.2f} seconds")

    def analyze(self):
        """Main method to analyze the PDF and return results."""
        self.page_paths = self.convert_pdf_pages_to_images()
        print(f"Starting to process {len(self.page_paths)} pages...")
        all_numbers, total_time, processing_time = self.process_pdf_pages()
        
        if all_numbers:
            max_page, max_value = max(all_numbers, key=lambda x: x[1])
            print(f"\nFinal Results:")
            print(f"Total numbers found: {len(all_numbers)}")
            print(f"Maximum value found: {max_value:,.0f} (on page {max_page})")
            
            print("\nTop 5 Largest Numbers:")
            top_5 = sorted(all_numbers, key=lambda x: x[1], reverse=True)[:5]
            for page, value in top_5:
                print(f"  - {value:,.0f} (on page {page})")
                
            print(f"\nTiming Statistics:")
            print(f"Total execution time: {total_time:.2f} seconds")
            print(f"Total processing time: {processing_time:.2f} seconds")
            print(f"Average time per page: {processing_time/len(self.page_paths):.2f} seconds")
        else:
            print("No numbers found in the document")
            
        return all_numbers

def main():
    pdf_path = "/home/ubuntu/my-conductor/FY25 Air Force Working Capital Fund.pdf"
    analyzer = PDFAnalyzer(pdf_path)
    analyzer.analyze()

if __name__ == "__main__":
    main()