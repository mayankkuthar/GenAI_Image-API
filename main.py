import google.generativeai as genai
import os
import json
import re
import logging
import base64
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image
import io
from typing import Optional, Dict, Any, List, Tuple
from datetime import datetime
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Document configurations
DAYBOOK_SHEET_OUTPUT_STRUCTURE = '''
{
  "Day 1": {
    "Date": "date",
    "Cash Sale": "float",
    "Credit Sale": "float",
    "Cash Collection": "float",
    "Cash Purchase": "float",
    "Credit Purchase": "float",
    "Cash Paid to Suppliers": "float",
    "Transportation": "float",
    "Owner's Wages": "float", 
    "Workers' Wages": "float",
    "Electricity": "float",
    "Repairs": "float",
    "Other Cost": "float",
    "Other Income": "float",
    "Loan": "float",
    "Interest": "float",
    "Amount": "float"
  }
}
'''

PT_SHEET_OLD_OUTPUT_STRUCTURE = '''
{
  "Period 1": {
    "Time Period Information": {
      "Start Date": "date",
      "End Date": "date"
    },
    "Fixed Particulars": {
      "Cash Sales": "float",
      "Credit Sales": "float",
      "Cash Received from Debtors": "float",
      "Cash Purchase": "float",
      "Credit Purchase": "float",
      "Cash Given to Creditors": "float",
      "Owner's Wages": "float",
      "Transportation": "float",
      "Room Rent": "float",
      "Worker's Salary or Wages": "float",
      "Electricity or Fuel": "float",
      "Repair and Maintenance": "float",
      "Other Cost": "float",
      "Other Income": "float",
      "Loan Repayment": "float",
      "Interest on Loan": "float",
      "Owner's Investment": "float"
    },
    "Summary Particulars": {
      "Closing Cash Balance": "float",
      "Closing Bank Balance": "float",
      "Purchase of Fixed Asset": "float",
      "Sale of Fixed Asset": "float",
      "Security Deposit Given": "float",
      "Security Deposit Returned": "float",
      "Loan Taken": "float",
      "Source of Loan": "string",
      "Closing Inventory": "float"
    }
  }
}
'''

PT_SHEET_NEW_OUTPUT_STRUCTURE = '''
{
  "Period 1": {
    "Time Period Information": {
      "Period From": "date",
      "Period To": "date"
    },
    "Details": {
      "Cash Sales": "float",
      "Credit Sales": "float",
      "Cash Received (from Customers)": "float",
      "Cash Purchase": "float",
      "Credit Purchase": "float",
      "Cash Paid to Suppliers": "float",
      "Owner's Wages": "float",
      "Workers' Wages": "float",
      "Transportation": "float",
      "Electricity": "float",
      "Repairs": "float",
      "Other Expenses": "float",
      "Interest Paid": "float",
      "Loan Repaid": "float",
      "Other Income": "float"
    },
    "Details on Capital": {
      "Fixed Assets Purchased": "float",
      "Fixed Assets Sold": "float",
      "Owner Reinvested": "float",
      "Owner's Withdrawals": "float",
      "Loan Taken": "float",
      "Source of Loan": "string",
      "Inventory (Goods / Stock)": "float"
    }
  }
}
'''

ONETIME_INFO_SHEET_OUTPUT_STRUCTURE = '''
{
  "Profile Information": {
    "Enterprise ID Number": "string",
    "Name of the Enterprise": "string",
    "Name of the Entrepreneur": "string",
    "Types of Enterprise": "string",
    "Date of Starting the Enterprise": "string",
    "New or Existing": "string",
    "Intervention Date by GUM": "string",
    "Mobile Number": "integer"
  },
  "Financial Information": {
    "Total Investment": "float",
    "Total Fixed Capital": "float",
    "Owner's Investment": "float",
    "Loan from SHG": "float",
    "Loan from Bank": "float",
    "Other Source": "float"
  },
  "Geography Information": {
    "Name of SHG": "string",
    "Name of VO": "string",
    "Name of CLF": "string",
    "Name of Village": "string",
    "Name of Panchayat": "string",
    "Name of Block": "string"
  },
  "Gram Vikas Internal Information": {
    "Name of GUM": "string",
    "GUM Mobile Number": "integer",
    "Date of Submission": "string",
    "Verified By": "string"
  }
}
'''

DOCUMENT_CONFIG = {
    "PT Sheet Old": {
        "output_structure": PT_SHEET_OLD_OUTPUT_STRUCTURE,
    },
    "PT Sheet New": {
        "output_structure": PT_SHEET_NEW_OUTPUT_STRUCTURE,
    },
    "Daybook": {
        "output_structure": DAYBOOK_SHEET_OUTPUT_STRUCTURE,
    },
    "One Time Info Sheet": {
        "output_structure": ONETIME_INFO_SHEET_OUTPUT_STRUCTURE,
    }
}

BASE_SYSTEM_MESSAGE_COMMON = """
You are an advanced AI system specialized in analyzing enterprise financial document images and extracting specific information. Your task is to carefully examine the provided image and extract data according to a given output structure. This task requires you to handle various financial document types, including invoices, receipts, bank statements, financial reports, and handwritten or semi-structured forms.

First, review the output structure that specifies the information you need to extract:

<output_structure>
{{OUTPUT_STRUCTURE}}
</output_structure>

This output structure is in JSON format. Each key represents a piece of financial information to look for in the image, and the corresponding value indicates the expected data type.

Instructions:
1. Analyze the financial document image thoroughly.
2. Pay special attention to the alignment of columns and rows, especially if the image appears tilted or angled.
3. Extract all relevant information as specified in the output structure.
4. Double-check the correspondence between particulars and their associated values to ensure accurate matching.
5. Format your response as a JSON object that exactly matches the given output structure.
6. Use appropriate data types (strings, numbers, booleans, null) for each field.
7. If information for a field is not present in the image, use null for numbers and booleans, or an empty string for strings.
8. If any content is unreadable, use "##UNREADABLE##" in place of that content.
9. Include all specified fields, even if some are not present in the image.

Remember:
- Pay special attention to handwriting interpretation. If unsure about a value, mark it "##UNREADABLE##".
- Only output the JSON data. Do not include any explanations or additional text
- Adhere strictly to the provided output structure.
- Use appropriate data types (strings, numbers, booleans, null) for each field
- Include all specified fields, even if some information is not present or illegible in the image.

Now, proceed with your examination and data extraction from the provided enterprise financial document image.
"""

# Initialize environment variables
load_dotenv()

app = FastAPI(title="Image Processing API")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Initialize Gemini model
GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("GOOGLE_API_KEY not found in environment variables!")

genai.configure(api_key=GEMINI_API_KEY)
gemini_model = genai.GenerativeModel('gemini-2.5-flash')

class JsonCorrector:
    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def extract_nested_json(self, text: str) -> str:
        """Extract the most complete JSON structure from text."""
        start = text.find('{')
        if start == -1:
            return text
        
        depth = 0
        in_string = False
        escape_char = False
        
        for i, char in enumerate(text[start:], start):
            if char == '\\' and not escape_char:
                escape_char = True
                continue
            
            if char == '"' and not escape_char:
                in_string = not in_string
            
            if not in_string:
                if char == '{':
                    depth += 1
                elif char == '}':
                    depth -= 1
                    if depth == 0:
                        return text[start:i+1]
            
            escape_char = False
        
        return text[start:]

    def balance_braces(self, text: str) -> str:
        """Balance opening and closing braces in the JSON string."""
        text = self.extract_nested_json(text)
        open_count = text.count('{')
        close_count = text.count('}')
        
        if open_count > close_count:
            text += '}' * (open_count - close_count)
        elif close_count > open_count:
            text = '{' * (close_count - open_count) + text
            
        return text

    def fix_trailing_commas(self, text: str) -> str:
        """Remove trailing commas in objects and arrays."""
        text = re.sub(r',(\s*})', r'\1', text)
        text = re.sub(r',(\s*])', r'\1', text)
        return text

    def add_missing_quotes(self, text: str) -> str:
        """Add missing quotes around property names."""
        def replace_unquoted(match):
            return f'"{match.group(1)}":'
        return re.sub(r'([a-zA-Z_][a-zA-Z0-9_]*)\s*:', replace_unquoted, text)

    def fix_missing_values(self, text: str) -> str:
        """Replace empty values with appropriate defaults."""
        text = re.sub(r':\s*,', ': "",', text)
        text = re.sub(r':\s*}', ': ""}', text)
        text = re.sub(r':\s*null\s*([,}])', ': ""\\1', text)
        text = re.sub(r'\[\s*([^]\}]*?)(?:\s*$|\s*})', r'[\1]', text)
        return text

    def correct_json(self, text: str) -> dict:
        """Attempt to correct malformed JSON and return a valid dictionary."""
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            try:
                corrected = text
                corrected = self.extract_nested_json(corrected)
                corrected = self.balance_braces(corrected)
                corrected = self.fix_trailing_commas(corrected)
                corrected = self.add_missing_quotes(corrected)
                corrected = self.fix_missing_values(corrected)
                
                return json.loads(corrected)
            except json.JSONDecodeError as e:
                try:
                    # Extract key-value pairs for partial structure
                    partial_structure = {}
                    pattern = r'"([^"]+)"\s*:\s*(?:"([^"]*)"|\[([^\]]*)\]|(\d+\.?\d*)|(\{[^}]*\})|([^,}\]]*))(?:,|\}|\]|$)'
                    matches = re.finditer(pattern, corrected)
                    
                    for match in matches:
                        key = match.group(1)
                        value = next((v for v in match.groups()[1:] if v is not None), "")
                        
                        try:
                            if re.match(r'^\d+\.?\d*$', value):
                                value = float(value)
                            elif value.startswith('[') or value.startswith('{'):
                                try:
                                    value = json.loads(value)
                                except:
                                    pass
                        except:
                            pass
                        
                        current = partial_structure
                        key_parts = key.split('.')
                        for part in key_parts[:-1]:
                            if part not in current:
                                current[part] = {}
                            current = current[part]
                        current[key_parts[-1]] = value
                    
                    if partial_structure:
                        return partial_structure
                except Exception as e:
                    return {
                        "error": "JSON parsing failed",
                        "partial_content": text[:2000] + "..." if len(text) > 2000 else text
                    }
                return {
                    "error": "JSON parsing failed",
                    "partial_content": text[:2000] + "..." if len(text) > 2000 else text
                }

@app.get("/")
def read_root():
    return {
        "message": "Welcome to the Image Processing API",
        "endpoints": {
            "/": "This welcome message",
            "/process-image/": "POST endpoint to process an image and extract information"
        }
    }

class ImageProcessor:
    def __init__(self, model_type="gemini"):
        """Initialize the image processor with API key."""
        self.model_type = model_type.lower()
        self.gemini_model = gemini_model
        self.json_corrector = JsonCorrector()

    async def process_with_gemini(self, image_data: bytes, system_message: str) -> Dict:
        """Process an image with Gemini model."""
        try:
            image = Image.open(io.BytesIO(image_data))
            img_byte_arr = io.BytesIO()
            image.save(img_byte_arr, format=image.format or 'PNG')
            img_byte_arr = img_byte_arr.getvalue()

            response = self.gemini_model.generate_content(
                contents=[
                    system_message,
                    {"mime_type": f"image/{image.format.lower() if image.format else 'png'}", 
                     "data": img_byte_arr}
                ],
                stream=False,
                generation_config={"temperature": 0.5, "max_output_tokens": 4096}
            )

            # Get response text and ensure it's complete
            response.resolve()
            
            # Clean and parse the response
            return self.json_corrector.correct_json(response.text)

        except Exception as e:
            error_msg = f"Error processing image with Gemini: {str(e)}"
            logging.error(error_msg)
            return {"error": error_msg}

    async def process_single_image(self, image_data: bytes, document_type: Optional[str] = None) -> Dict:
        """Process a single image using the configured model."""
        try:
            # Get the appropriate output structure based on document type
            output_structure = None
            if document_type:
                if document_type in DOCUMENT_CONFIG:
                    output_structure = DOCUMENT_CONFIG[document_type]["output_structure"]
                else:
                    raise ValueError(f"Invalid document type. Supported types are: {list(DOCUMENT_CONFIG.keys())}")

            # Prepare system message
            system_message = BASE_SYSTEM_MESSAGE_COMMON
            if output_structure:
                system_message = system_message.replace("{{OUTPUT_STRUCTURE}}", output_structure)

            # Process with Gemini
            return await self.process_with_gemini(image_data, system_message)

        except Exception as e:
            error_msg = f"Error processing image: {str(e)}"
            logging.error(error_msg)
            return {"error": error_msg}

# Initialize the image processor
image_processor = ImageProcessor()

@app.post("/process-image/")
async def process_image(
    file: UploadFile = File(...), 
    document_type: Optional[str] = None
):
    """
    Process an uploaded image and extract information based on document type.
    
    Args:
        file: The uploaded image file
        document_type: Optional document type ("PT Sheet Old", "PT Sheet New", "Daybook", "One Time Info Sheet")
    """
    try:
        # Read and validate the image file
        image_data = await file.read()
        try:
            # Validate that it's a valid image file
            Image.open(io.BytesIO(image_data))
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid image file: {str(e)}")

        # Process the image
        result = await image_processor.process_single_image(image_data, document_type)

        if "error" in result:
            raise HTTPException(status_code=500, detail=result["error"])

        return JSONResponse(content=result)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))