'''
Created by Satya Suman Shu
Created on 7 DEC SAT 11:01:37 AM 2024

'''
import re
import requests
from transformers import DonutProcessor, VisionEncoderDecoderModel
import torch
import json
from PIL import Image
from io import BytesIO
import time
import matplotlib.pyplot as plt

class ReceiptProcessor:
    def __init__(self, model_name="naver-clova-ix/donut-base-finetuned-cord-v2"):
        self.processor = DonutProcessor.from_pretrained(model_name)
        self.model = VisionEncoderDecoderModel.from_pretrained(model_name)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)

    def process_image(self, image_path, filename):
        """Processes a single image and extracts relevant fields."""
        timings = {}

        # Load image
        start_time = time.time()
        try:
            image = Image.open(image_path)
        except Exception as e:
            print(f"Error loading image {filename}: {e}")
            return None, timings
        timings['Load Image'] = time.time() - start_time

        # Prepare input
        start_time = time.time()
        task_prompt = "<s_cord-v2>"
        decoder_input_ids = self.processor.tokenizer(task_prompt, add_special_tokens=False, return_tensors="pt").input_ids
        pixel_values = self.processor(image, return_tensors="pt").pixel_values.to(self.device)
        timings['Prepare Input'] = time.time() - start_time

        # Model inference
        start_time = time.time()
        try:
            outputs = self.model.generate(
                pixel_values,
                decoder_input_ids=decoder_input_ids.to(self.device),
                max_length=self.model.decoder.config.max_position_embeddings,
                pad_token_id=self.processor.tokenizer.pad_token_id,
                eos_token_id=self.processor.tokenizer.eos_token_id,
                use_cache=True,
                bad_words_ids=[[self.processor.tokenizer.unk_token_id]],
                return_dict_in_generate=True,
            )
        except Exception as e:
            print(f"Error during model inference for {filename}: {e}")
            return None, timings
        timings['Model Inference'] = time.time() - start_time

        # Decode and format output
        start_time = time.time()
        sequence = self.processor.batch_decode(outputs.sequences)[0]
        sequence = sequence.replace(self.processor.tokenizer.eos_token, "").replace(self.processor.tokenizer.pad_token, "")
        sequence = re.sub(r"<.*?>", "", sequence, count=1).strip()
        timings['Decode Output'] = time.time() - start_time

        # Extract and format data
        start_time = time.time()
        try:
            extracted_data = self.processor.token2json(sequence)
            formatted_data = self.format_output(extracted_data, filename)
        except Exception as e:
            print(f"Error processing extracted data for {filename}: {e}")
            return None, timings
        timings['Format Output'] = time.time() - start_time

        return formatted_data, timings

    def format_output(self, data, filename):
        """Formats the extracted data into the desired JSON structure."""
        return {
            "company_name": data.get("menu", [{}])[0].get("nm", "Unknown"),
            "company_registration": data.get("menu", [{}])[0].get("unitprice", "Unknown"),
            "invoice_type": data.get("menu", [{}])[1].get("nm", "Unknown"),
            "date": data.get("menu", [{}])[1].get("price", "Unknown"),
            "user_time": data.get("menu", [{}])[2].get("price", "Unknown"),
            "items": [
                {
                    "item_name": item.get("nm", "Unknown"),
                    "unit_price": item.get("unitprice", "Unknown"),
                    "quantity": item.get("cnt", "Unknown"),
                    "price": item.get("price", "Unknown")
                }
                for item in data.get("menu", [])[3:]
            ],
            "subtotal": data.get("sub_total", {}).get("subtotal_price", "Unknown"),
            "discount": data.get("sub_total", {}).get("discount_price", "Unknown"),
            "tax": data.get("sub_total", {}).get("tax_price", "Unknown"),
            "total_price": data.get("total", {}).get("total_price", "Unknown"),
            "cash_price": data.get("total", {}).get("cashprice", "Unknown"),
            "change_price": data.get("total", {}).get("changeprice", "Unknown"),
            "credit_card_price": data.get("total", {}).get("creditcardprice", "Unknown"),
            "filename": filename
        }

    def process_images_from_url(self, url_list):
        """Processes multiple images from given URLs and returns extracted data."""
        results = []
        for url in url_list:
            try:
                response = requests.get(url)
                if response.status_code == 200:
                    image = BytesIO(response.content)
                    filename = url.split("/")[-1]
                    result, timings = self.process_image(image, filename)
                    if result:
                        results.append((result, timings))
                else:
                    print(f"Failed to download image from {url}")
            except Exception as e:
                print(f"Error downloading image from {url}: {e}")
        return results

if __name__ == "__main__":
    # URLs of images
    image_urls = [
        "https://raw.githubusercontent.com/zzzDavid/ICDAR-2019-SROIE/master/data/img/001.jpg",
        "https://raw.githubusercontent.com/zzzDavid/ICDAR-2019-SROIE/master/data/img/002.jpg",
        "https://raw.githubusercontent.com/zzzDavid/ICDAR-2019-SROIE/master/data/img/003.jpg",
        "https://raw.githubusercontent.com/zzzDavid/ICDAR-2019-SROIE/master/data/img/004.jpg",
        "https://raw.githubusercontent.com/zzzDavid/ICDAR-2019-SROIE/master/data/img/005.jpg"
    ]

    processor = ReceiptProcessor()
    extracted_data = processor.process_images_from_url(image_urls)

    # Plot timings for each image
    for result, timings in extracted_data:
        filename = result["filename"]
        plt.bar(timings.keys(), timings.values())
        plt.xlabel("Processing Steps")
        plt.ylabel("Time Taken (s)")
        plt.title(f"Processing Time for {filename}")
        plt.xticks(rotation=45)
        plt.show()

    # Print results
    for data, timings in extracted_data:
        print(json.dumps(data, indent=4))
