from google import genai
from google.genai import types
import os
import json
import re
import logging
import base64
from enum import Enum
from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image, ImageOps
import io
from typing import Optional, Dict, Any, List, Tuple, Annotated
from datetime import datetime
from dotenv import load_dotenv

# Max upload size (bytes) - Vercel Hobby=4.5MB, Pro=25MB
MAX_UPLOAD_SIZE = 4.5 * 1024 * 1024  # 4.5 MB default for Hobby plan
# Target max dimension for Gemini (reduces file size while keeping quality)
MAX_IMAGE_DIMENSION = 2048
# JPEG quality for compression
JPEG_QUALITY = 85


class DocumentType(str, Enum):
    """Document types supported by the API."""
    PT_SHEET_OLD = "PT Sheet Old"
    PT_SHEET_NEW = "PT Sheet New"
    DAYBOOK = "Daybook"
    ONE_TIME_INFO_SHEET = "One Time Info Sheet"

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Document configurations
DAYBOOK_SHEET_OUTPUT_STRUCTURE = '''
{
  "Daybook Record": {
    "Enterprise ID": "string",
    "Month": "MMM YY",
    "Day 1": {
        "Sale in Cash and UPI": "float",
        "Sale on Credit": "float",
        "Cash Received": "float",
        "Raw Material (Cash & UPI)": "float",
        "Raw Material (Credit)": "float",
        "Cash Paid to Suppliers": "float",
        "Transportation Charges": "float",
        "Worker Wages": "float",
        "Owner Salary": "float",
        "Other Owner Withdrawal": "float",
        "Electricity / Fuel": "float",
        "Shop Rent paid": "float",
        "Loan Repayment": "float",
        "Other Income": "float",
        "Other costs": "float"
    }
  }
}
'''

DAY_FIELDS = [
    "Sale in Cash and UPI",
    "Sale on Credit",
    "Cash Received",
    "Raw Material (Cash & UPI)",
    "Raw Material (Credit)",
    "Cash Paid to Suppliers",
    "Transportation Charges",
    "Worker Wages",
    "Owner Salary",
    "Other Owner Withdrawal",
    "Electricity / Fuel",
    "Shop Rent paid",
    "Loan Repayment",
    "Other Income",
    "Other costs"
]

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

BASE_SYSTEM_MESSAGE_FOR_DAYBOOK = """
You are a high-precision financial data extraction AI. Your task is to digitize "Daybook" images into structured JSON with MINIMAL output size.

### INPUT CONTEXT
You will receive an image of a pre-printed financial form.
* **Layout:** The document is a grid table. Rows (Vertical) = Days (1-31). Columns (Horizontal) = Categories.
* **Language:** Headers are bilingual (English/Odia). Use the English text.

### OUTPUT SCHEMA
Extract data strictly according to the following JSON structure.
**Critical:** The schema below defines the structure for a single day. You must generate keys for every day present in the image that contains data.

<output_structure>
{{OUTPUT_STRUCTURE}}
</output_structure>

### EXTRACTION INSTRUCTIONS
1.  **Anchors:** Locate Enterprise ID (Top Left) and Month (Top Right).
2.  **Grid Traversal:** Scan the leftmost column (Day 1-31). For every row with handwritten data, create a "Day X" key.
3.  **Data Type:** Extract integers only from cells with visible handwritten numbers.
4.  **CRITICAL OUTPUT RULE - SPARSE JSON (MANDATORY):**
    - ONLY output fields that contain actual numerical data
    - NEVER output a field if the cell is blank, empty, illegible, contains a dash (-), or has no written value
    - NEVER include null values in your output - completely omit the field instead
    - NEVER include fields with null, None, or empty values - omit them entirely
    - Each "Day X" object should ONLY contain the 2-5 fields that actually have data
    - DO NOT include all 14 fields for each day - only include fields with actual values
    - This is NOT optional - sparse output is mandatory to reduce token usage
    - If you include null values, your output is INCORRECT
5.  **Formatting:** Output ONLY valid JSON with no markdown, no comments, no explanations.

### CORRECT OUTPUT EXAMPLE
If Day 1 has values in only 2 cells and Day 2 has values in only 3 cells:

{
  "Daybook Record": {
    "Enterprise ID": "ABC123",
    "Month": "Jan 25", // Do not use full month/year names. Keep it 3 chars of Month, a space and 2 digits of the year.
    "Day 1": {
      "Sale in Cash and UPI": 500,
      "Shop Rent paid": 200
    },
    "Day 2": {
      "Sale in Cash and UPI": 300,
      "Raw Material (Cash & UPI)": 150,
      "Other costs": 50
    }
  }
}

### INCORRECT OUTPUT (DO NOT DO THIS)
{
  "Daybook Record": {
    "Enterprise ID": "ABC123",
    "Month": "Jan 25",
    "Day 1": {
      "Sale in Cash and UPI": 500,
      "Sale on Credit": null,
      "Cash Received": null,
      ...ALL OTHER FIELDS AS NULL...
      "Shop Rent paid": 200
    }
  }
}
"""

# Initialize environment variables
load_dotenv()

app = FastAPI(
    title="GenAI Image Processing API",
    description="""
## Financial Document Processing API

Upload images of handwritten financial documents and extract structured data using AI.

### Supported Document Types
- **PT Sheet Old** - Periodic Transaction Sheet (Old Format)
- **PT Sheet New** - Periodic Transaction Sheet (New Format)  
- **Daybook** - Daily financial transaction records
- **One Time Info Sheet** - Enterprise profile and financial information

### How to Use
1. Select a document type from the dropdown
2. Upload an image file (JPEG/PNG, max 4.5MB)
3. Receive structured JSON data extracted from the image

### Notes
- Images are auto-compressed if they exceed size limits
- Mobile photos are auto-rotated based on EXIF data
- For best results, ensure the document is clearly visible and well-lit
    """,
    version="1.0.0",
    contact={
        "name": "API Support",
    }
)

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

gemini_client = genai.Client(api_key=GEMINI_API_KEY)

# Model + thinking level are configurable via env so they can change without a code edit.
# Defaults chosen by evaluation on the sample Daybook (2026-11-20): gemini-3.8-flash @ LOW thinking
# gave 88-91/95 cells correct in ~12 s. gemini-2.5-flash is deprecated from 2026-12-01.
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")
GEMINI_THINKING_LEVEL = os.getenv("GEMINI_THINKING_LEVEL", "LOW")
# Thinking tokens count toward this limit. 4096 truncated replies mid-sheet; 16384 leaves ample room.
GEMINI_MAX_OUTPUT_TOKENS = 16384

# Orientation pre-check: a fast model reports which image edge is the top of the page, and the image is
# rotated upright BEFORE extraction. Evaluated 2026-11-26: on a sideways photo the extractor shifted values
# one day down and misread the year; prompting it to "mentally rotate" made this worse, while rotating the
# pixels fixed it. gemini-3.6-flash @ MINIMAL detected all 4 orientations correctly in ~2.5-3.7 s.
# Set GEMINI_ORIENTATION_MODEL to an empty string to disable the check.
GEMINI_ORIENTATION_MODEL = os.getenv("GEMINI_ORIENTATION_MODEL", "gemini-3.6-flash")
GEMINI_ORIENTATION_THINKING_LEVEL = os.getenv("GEMINI_ORIENTATION_THINKING_LEVEL", "MINIMAL")
ORIENTATION_CHECK_MAX_SIDE = 768  # the check only needs the layout, so send a small copy (faster)
ORIENTATION_PROMPT = (
    'This is a photo of a printed form or ledger page. Its printed header (e.g. "Enterprise ID", "Month", '
    '"Year", or the form title) is at the TOP of the page. Which edge of the IMAGE is the top of the page on? '
    'Answer JSON: {"page_top_edge": "top" | "right" | "bottom" | "left"}'
)
# Counter-clockwise degrees for PIL's Image.rotate that move each edge to the top.
ROTATION_FOR_TOP_EDGE = {"top": 0, "right": 90, "bottom": 180, "left": 270}

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

@app.get(
    "/",
    summary="API Health Check",
    description="Returns API information and available endpoints."
)
def read_root():
    return {
        "message": "GenAI Image Processing API",
        "version": "1.0.0",
        "endpoints": {
            "/": "This welcome message",
            "/process-image/": "POST - Process financial document images",
            "/docs": "Interactive API documentation (Swagger UI)",
            "/redoc": "Alternative API documentation (ReDoc)"
        },
        "supported_document_types": [dt.value for dt in DocumentType]
    }

def compress_image(image_data: bytes) -> bytes:
    """
    Compress and resize image to fit within Vercel's payload limits.
    Mobile cameras produce 3-10+ MB images that exceed Vercel's 4.5 MB limit.
    """
    try:
        img = Image.open(io.BytesIO(image_data))
        
        # Convert RGBA to RGB (JPEG doesn't support alpha)
        if img.mode in ('RGBA', 'LA', 'P'):
            background = Image.new('RGB', img.size, (255, 255, 255))
            if img.mode == 'P':
                img = img.convert('RGBA')
            background.paste(img, mask=img.split()[-1] if 'A' in img.mode else None)
            img = background
        
        # Auto-rotate based on EXIF data (mobile photos)
        try:
            img = ImageOps.exif_transpose(img)
        except:
            pass
        
        # Resize if too large (maintains aspect ratio)
        original_size = img.size
        if max(img.size) > MAX_IMAGE_DIMENSION:
            ratio = MAX_IMAGE_DIMENSION / max(img.size)
            new_size = (int(img.size[0] * ratio), int(img.size[1] * ratio))
            img = img.resize(new_size, Image.LANCZOS)
            logging.info(f"Resized image from {original_size} to {new_size}")
        
        # Convert to JPEG and compress
        img_byte_arr = io.BytesIO()
        
        # If originally PNG with transparency, keep as PNG but reduce quality
        if img.mode == 'RGBA':
            img.save(img_byte_arr, format='PNG', optimize=True)
        else:
            img.save(img_byte_arr, format='JPEG', quality=JPEG_QUALITY, optimize=True)
        
        compressed = img_byte_arr.getvalue()
        
        original_mb = len(image_data) / (1024 * 1024)
        compressed_mb = len(compressed) / (1024 * 1024)
        
        if original_mb > compressed_mb:
            logging.info(f"Compressed image: {original_mb:.2f}MB -> {compressed_mb:.2f}MB")
        
        return compressed
        
    except Exception as e:
        logging.error(f"Error compressing image: {str(e)}")
        # Return original if compression fails
        return image_data


def _encode_jpeg(img: Image.Image) -> bytes:
    """Encode as high-quality JPEG (used only when the pixels had to change)."""
    if img.mode != "RGB":
        img = img.convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=95)
    return buf.getvalue()


def detect_page_top_edge(image_data: bytes) -> Optional[str]:
    """Ask a fast model which image edge is the top of the page. Returns None if unknown."""
    if not GEMINI_ORIENTATION_MODEL:
        return None
    try:
        preview = Image.open(io.BytesIO(image_data))
        preview.thumbnail((ORIENTATION_CHECK_MAX_SIDE, ORIENTATION_CHECK_MAX_SIDE))
        started = datetime.now()
        response = gemini_client.models.generate_content(
            model=GEMINI_ORIENTATION_MODEL,
            contents=[types.Part.from_bytes(data=_encode_jpeg(preview), mime_type="image/jpeg"), ORIENTATION_PROMPT],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                max_output_tokens=1024,  # thinking tokens count toward this, so keep headroom
                thinking_config=types.ThinkingConfig(thinking_level=GEMINI_ORIENTATION_THINKING_LEVEL),
            ),
        )
        parsed = json.loads(response.text or "{}")
        edge = parsed.get("page_top_edge") if isinstance(parsed, dict) else None
        logging.info(f"Orientation check ({GEMINI_ORIENTATION_MODEL}): page top = {edge!r}, "
                     f"{(datetime.now() - started).total_seconds():.1f}s")
        return edge if edge in ROTATION_FOR_TOP_EDGE else None
    except Exception as e:
        logging.warning(f"Orientation check failed, extracting from the image as-is: {e}")
        return None


def make_upright(image_data: bytes) -> bytes:
    """
    Return the image rotated so the page is upright. Any failure returns the image unchanged,
    so the orientation check can never block an extraction.
    """
    try:
        img = Image.open(io.BytesIO(image_data))
        # Apply a phone's EXIF rotation tag first, so the check and the extractor see the same pixels.
        if img.getexif().get(0x0112, 1) not in (None, 1):
            image_data = _encode_jpeg(ImageOps.exif_transpose(img))

        edge = detect_page_top_edge(image_data)
        degrees = ROTATION_FOR_TOP_EDGE.get(edge, 0) if edge else 0
        if degrees == 0:
            return image_data
        logging.info(f"Page top is on the {edge} edge: rotating {degrees} degrees to upright")
        return _encode_jpeg(Image.open(io.BytesIO(image_data)).rotate(degrees, expand=True))
    except Exception as e:
        logging.warning(f"Could not make image upright, using it as-is: {e}")
        return image_data


class ImageProcessor:
    def __init__(self, model_type="gemini"):
        """Initialize the image processor with API key."""
        self.model_type = model_type.lower()
        self.gemini_client = gemini_client
        self.json_corrector = JsonCorrector()
        
    async def fill_missing_fields(self, daybook_response: Dict[str, Any]) -> Dict[str, Any]:
        """
        Post-processing for Daybook:
        1. Ensure all days 1–31 are present
        2. Ensure all DAY_FIELDS exist for each day
        3. Preserve non-day keys (Month, Year, Enterprise ID, etc.)
        4. Maintain exact DAY_FIELDS order
        """

        if not isinstance(daybook_response, dict):
            return daybook_response

        if "Daybook Record" not in daybook_response:
            return daybook_response

        record = daybook_response["Daybook Record"]

        # New ordered record
        new_record = {}

        # Preserve non-day keys
        for key, value in record.items():
            if not key.startswith("Day"):
                new_record[key] = value

        # Ensure Day 1 to Day 31
        for day_num in range(1, 32):
            day_key = f"Day {day_num}"

            if day_key in record and isinstance(record[day_key], dict):
                day_data = record[day_key]
                ordered_day_data = {}

                # Ensure all DAY_FIELDS exist and are ordered
                for field in DAY_FIELDS:
                    ordered_day_data[field] = day_data.get(field, None)

                new_record[day_key] = ordered_day_data
            else:
                # Missing day → fill with None fields
                new_record[day_key] = {field: None for field in DAY_FIELDS}

        daybook_response["Daybook Record"] = new_record
        return daybook_response

    async def process_with_gemini(self, image_data: bytes, system_message: str) -> Dict:
        """Process an image with Gemini model."""
        try:
            # Send the original bytes (re-encoding a JPEG would lose quality); only detect the format.
            image_format = (Image.open(io.BytesIO(image_data)).format or "PNG").upper()
            mime_type = "image/jpeg" if image_format in ("JPEG", "MPO") else f"image/{image_format.lower()}"

            started = datetime.now()
            response = self.gemini_client.models.generate_content(
                model=GEMINI_MODEL,
                contents=[
                    types.Part.from_bytes(data=image_data, mime_type=mime_type),
                    system_message,
                ],
                # No temperature: deprecated (ignored) for Gemini 3.6+ models.
                config=types.GenerateContentConfig(
                    max_output_tokens=GEMINI_MAX_OUTPUT_TOKENS,
                    response_mime_type="application/json",  # forces valid JSON: no ``` fences, no // comments
                    thinking_config=types.ThinkingConfig(thinking_level=GEMINI_THINKING_LEVEL),
                ),
            )

            candidate = response.candidates[0] if response.candidates else None
            finish_reason = getattr(getattr(candidate, "finish_reason", None), "name", "UNKNOWN")
            usage = response.usage_metadata
            logging.info(
                f"Gemini {GEMINI_MODEL} ({GEMINI_THINKING_LEVEL}): {(datetime.now() - started).total_seconds():.1f}s, "
                f"finish={finish_reason}, thinking_tokens={getattr(usage, 'thoughts_token_count', None)}, "
                f"answer_tokens={getattr(usage, 'candidates_token_count', None)}"
            )

            # Clean and parse the response
            return self.json_corrector.correct_json(response.text or "")

        except Exception as e:
            error_msg = f"Error processing image with Gemini: {str(e)}"
            logging.error(error_msg)
            return {"error": error_msg}
    
    async def process_single_image(self, image_data: bytes, document_type: Optional[str] = None) -> Dict:
        """Process a single image using the configured model."""        
        try:
            output_structure = None

            if document_type:
                if document_type not in DOCUMENT_CONFIG:
                    raise ValueError(
                        f"Invalid document type. Supported types are: {list(DOCUMENT_CONFIG.keys())}"
                    )
                output_structure = DOCUMENT_CONFIG[document_type]["output_structure"]

            # 🔹 DEFAULT system message
            system_message = BASE_SYSTEM_MESSAGE_COMMON

            # 🔹 CONDITION: If Daybook → use Daybook-specific system message
            if document_type == "Daybook":
                system_message = BASE_SYSTEM_MESSAGE_FOR_DAYBOOK

            if output_structure:
                system_message = system_message.replace(
                    "{{OUTPUT_STRUCTURE}}",
                    output_structure
                )

            # Rotate sideways/upside-down photos upright first: extraction on rotated grids shifts rows
            image_data = make_upright(image_data)

            # Process with Gemini
            result = await self.process_with_gemini(image_data, system_message)

            #  Apply fill_missing_fields ONLY for Daybook
            if document_type == "Daybook" and isinstance(result, dict):
                result = await self.fill_missing_fields(result)

            return result
        
        except Exception as e:
            error_msg = f"Error processing image: {str(e)}"
            logging.error(error_msg)
            return {"error": error_msg}

# Initialize the image processor
image_processor = ImageProcessor()

@app.post(
    "/process-image/",
    summary="Process financial document image",
    description="""
Upload an image of a financial document to extract structured data.

**Supported formats:** JPEG, PNG
**Max file size:** 4.5MB (auto-compressed if larger)

The API will:
1. Validate and compress the image if needed
2. Send to Gemini AI for analysis
3. Return structured JSON based on document type
    """,
    response_description="Structured JSON data extracted from the image"
)
async def process_image(
    file: UploadFile = File(
        ..., 
        description="Image file to process (JPEG/PNG, max 4.5MB)",
        media_type="image/jpeg,image/png"
    ), 
    document_type: Optional[DocumentType] = Form(
        None,
        description="Type of document to process. Select from dropdown."
    )
):
    try:
        # Read the image file
        image_data = await file.read()
        
        # Check file size before processing
        file_size_mb = len(image_data) / (1024 * 1024)
        logging.info(f"Received image: {file.filename}, Size: {file_size_mb:.2f}MB, Type: {document_type}")
        
        # Validate it's a valid image first
        try:
            Image.open(io.BytesIO(image_data))
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid image file: {str(e)}")
        
        # Compress image if it's too large for Vercel
        if len(image_data) > MAX_UPLOAD_SIZE:
            logging.warning(f"Image exceeds {MAX_UPLOAD_SIZE/(1024*1024):.1f}MB limit, compressing...")
            image_data = compress_image(image_data)
            
            # If still too large after compression, reject
            if len(image_data) > MAX_UPLOAD_SIZE:
                compressed_mb = len(image_data) / (1024 * 1024)
                raise HTTPException(
                    status_code=413,
                    detail=f"Image too large ({compressed_mb:.1f}MB). Maximum allowed is {MAX_UPLOAD_SIZE/(1024*1024):.1f}MB."
                )
        
        # Convert enum to string value for processing
        doc_type_str = document_type.value if document_type else None
        
        # Process the image
        result = await image_processor.process_single_image(image_data, doc_type_str)

        if "error" in result:
            raise HTTPException(status_code=500, detail=result["error"])

        return JSONResponse(content=result)

    except HTTPException:
        raise
    except Exception as e:
        error_msg = str(e)
        # Handle Vercel's payload too large error
        if "payload" in error_msg.lower() or "too large" in error_msg.lower():
            raise HTTPException(
                status_code=413,
                detail="Image too large. Please use a smaller image or compress it before uploading."
            )
        raise HTTPException(status_code=500, detail=error_msg)