# Donut-model-implementation


# Receipt Processor

## Overview
This project implements a `ReceiptProcessor` class that extracts structured data from receipt images using the [DonutProcessor](https://huggingface.co/docs/transformers/model_doc/donut) and [VisionEncoderDecoderModel](https://huggingface.co/docs/transformers/model_doc/vision_encoder_decoder) from Hugging Face's Transformers library. The application processes images either from local paths or URLs and outputs the extracted data in JSON format.

---

## Features
- Extracts and formats receipt data, including:
  - Company name and registration.
  - Invoice type and date.
  - Purchased items, including name, unit price, quantity, and total price.
  - Subtotal, discounts, tax, total price, and payment breakdown (cash, change, credit card).
- Supports local and URL-based image inputs.
- Measures and plots processing time for each step.

---

## Installation
### Prerequisites
1. Python 3.11.2
2. Install required libraries:
   ```bash
   pip install torch transformers pillow requests matplotlib
   ```
3. GPU support (optional):
   Ensure that [PyTorch](https://pytorch.org/get-started/locally/) is installed with CUDA support for faster processing.

---

## Usage
### 1. Initialize the Processor
The `ReceiptProcessor` class uses a pre-trained model. By default, it loads the `naver-clova-ix/donut-base-finetuned-cord-v2` model:
```python
processor = ReceiptProcessor()
```

### 2. Process Local Images
To process a local image:
```python
result, timings = processor.process_image("path_to_image.jpg", "image_filename")
print(result)
```

### 3. Process Images from URLs
To process multiple images from URLs:
```python
image_urls = [
    "https://example.com/image1.jpg",
    "https://example.com/image2.jpg"
]
extracted_data = processor.process_images_from_url(image_urls)
for data, timings in extracted_data:
    print(json.dumps(data, indent=4))
```

### 4. Plot Processing Times
Timing information for each processing step is available for visualization:
```python
for result, timings in extracted_data:
    filename = result["filename"]
    plt.bar(timings.keys(), timings.values())
    plt.xlabel("Processing Steps")
    plt.ylabel("Time Taken (s)")
    plt.title(f"Processing Time for {filename}")
    plt.xticks(rotation=45)
    plt.show()
```

---

## Code Structure
### `ReceiptProcessor`
- **Methods**:
  - `__init__(self, model_name)`: Initializes the model and processor.
  - `process_image(self, image_path, filename)`: Extracts and formats data from a single image.
  - `format_output(self, data, filename)`: Structures the extracted data into JSON format.
  - `process_images_from_url(self, url_list)`: Downloads and processes multiple images from URLs.

---

## Example Output
```json
{
    "company_name": "Example Store",
    "company_registration": "12345678",
    "invoice_type": "Retail",
    "date": "2024-12-07",
    "user_time": "11:00 AM",
    "items": [
        {
            "item_name": "Item A",
            "unit_price": "10.00",
            "quantity": "2",
            "price": "20.00"
        },
        {
            "item_name": "Item B",
            "unit_price": "5.00",
            "quantity": "1",
            "price": "5.00"
        }
    ],
    "subtotal": "25.00",
    "discount": "0.00",
    "tax": "2.50",
    "total_price": "27.50",
    "cash_price": "30.00",
    "change_price": "2.50",
    "credit_card_price": "0.00",
    "filename": "receipt.jpg"
}
```

---

## Notes
- Ensure receipt images are clear and readable for better accuracy.
- The output JSON structure may vary based on the quality and content of the receipt.
- Use GPUs to speed up the inference process.

---

## License
This project is licensed under the MIT License. See the `LICENSE` file for details.

---

## Acknowledgments
- [Hugging Face](https://huggingface.co/) for providing the pre-trained models and libraries.
- [ICDAR SROIE Dataset](https://github.com/zzzDavid/ICDAR-2019-SROIE) for example receipt images.

---

## Contact
For any issues or suggestions, please feel free to reach out to:
- **Author**: Satya Suman Shu
- **Created On**: 7 December 2024

