import os
import base64
import requests
import streamlit as st # Import streamlit first
import io # Needed for BytesIO

# --- Page Config ---
# ✅ MUST BE THE VERY FIRST STREAMLIT COMMAND, AND ONLY CALLED ONCE
st.set_page_config(page_title="🧠 AI Medical Scan Analyzer", layout="centered")

# --- Import other libraries AFTER page config ---
from PIL import Image
import torch
from transformers import (
    BlipProcessor,
    BlipForConditionalGeneration,
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    # AutoImageProcessor, # Removed - Not using HF ViT model for classification
    # AutoModelForImageClassification # Removed - Not using HF ViT model for classification
)


gemini_api_key = st.secrets["gemini"]["api_key"]

# --- Other Configurations (Non-Streamlit related first) ---
os.environ["STREAMLIT_WATCHER_TYPE"] = "none"
GEMINI_MODEL = "gemini-1.5-flash-latest" # Use a current vision model
GEMINI_API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent"

# --- Load Secrets (Streamlit command, comes AFTER set_page_config) ---
try:
    gemini_api_key= st.secrets["gemini_api_key"]
except FileNotFoundError:
    st.exception("⚠️ Error: Gemini API Key file (.streamlit/secrets.toml) not found.")
    st.warning("Please create the secrets file with your gemini_api_key.")
    st.stop()
except KeyError:
    st.exception("⚠️ Error: `gemini_api_key` not found in secrets.toml.")
    st.warning("Please add the `gemini_api_key = 'YourKey'` line to .streamlit/secrets.toml.")
    st.stop()


# --- Model Loading Functions (Definitions using st.cache_resource) ---

# Removed Hugging Face model loader as it's unsuitable for this validation task
# @st.cache_resource
# def load_hf_model():
#     processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")
#     model = AutoModelForImageClassification.from_pretrained("google/vit-base-patch16-224-in21k")
#     return processor, model

@st.cache_resource
def load_caption_model():
    """Loads the BLIP image captioning model and processor."""
    try:
        processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
        return processor, model
    except Exception as e:
        st.error(f"Fatal Error: Could not load captioning model.")
        st.exception(e)
        st.stop()

@st.cache_resource
def load_text_model():
    """Loads the FLAN-T5 text generation model and tokenizer."""
    try:
        tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
        model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")
        return tokenizer, model
    except Exception as e:
        st.error(f"Fatal Error: Could not load text generation model.")
        st.exception(e)
        st.stop()

# --- Load Models (Actual execution which involves Streamlit cache) ---
# hf_processor, hf_model = load_hf_model() # Removed call
caption_processor, caption_model = load_caption_model()
text_tokenizer, text_model = load_text_model()

# --- Helper Functions ---

# Removed is_medical_scan_hf function - ViT model is not appropriate
# def is_medical_scan_hf(image):
#     ...

def is_medical_scan_gemini(image_bytes):
    """Checks if the image is likely a medical scan using Gemini Vision."""
    if not gemini_api_key:
        st.error("Gemini API Key is not configured properly.")
        return False

    headers = {"Content-Type": "application/json"}
    img_b64 = base64.b64encode(image_bytes).decode()

    # Using the recommended payload structure with text prompt
    data = {
        "contents": [{
            "parts": [
                {"text": "Is this image a medical scan (like X-ray, CT, MRI, ultrasound)? Answer with only 'Yes' or 'No'."},
                {"inline_data": {"mime_type": "image/jpeg", "data": img_b64}} # Mime type might need adjustment based on actual upload
            ]
        }],
        "generationConfig": {
             "temperature": 0.2,
             "maxOutputTokens": 5
        }
    }

    try:
        response = requests.post(
            f"{GEMINI_API_URL}?key={gemini_api_key}",
            headers=headers,
            json=data,
            timeout=60
        )
        response.raise_for_status()
        response_data = response.json()

        # Safely extract text response
        try:
            resp_text = response_data['candidates'][0]['content']['parts'][0]['text'].strip().lower()
            # st.write(f"Gemini Raw Validation Response: '{resp_text}'") # Uncomment for DEBUG
            return resp_text.startswith("yes")
        except (KeyError, IndexError, TypeError) as e:
            st.error(f"Error parsing Gemini response structure: {e}")
            st.json(response_data)
            return False

    except requests.exceptions.HTTPError as http_err:
        st.error(f"Gemini API HTTP Error: {http_err}")
        try:
            error_details = http_err.response.json()
            st.json(error_details)
            if "API key not valid" in str(error_details):
                 st.warning("Please check your GEMINI_API_KEY in secrets.toml.")
            elif "permission" in str(error_details).lower():
                 st.warning("Ensure the Gemini API is enabled for your project and the key has permissions.")
        except Exception:
             st.error(f"Response body: {http_err.response.text}")
        return False
    except requests.exceptions.RequestException as req_err:
        st.error(f"Gemini API Request Failed: {req_err}")
        return False
    except Exception as e:
        st.error(f"An unexpected error occurred during Gemini validation.")
        st.exception(e)
        return False


# --- Streamlit UI Elements ---
st.title("🧠 Medical Scan Analyzer")
st.markdown("Upload a medical scan (X-ray, CT, MRI, etc.). This AI will attempt to describe it and suggest potential next steps based on the description.")
st.warning("⚠️ **Disclaimer:** This tool is for informational purposes only and is not a substitute for professional medical advice, diagnosis, or treatment.")

uploaded_file = st.file_uploader(
    "📤 Upload a medical scan",
    type=["jpg", "jpeg", "png", "bmp", "tif", "tiff"],
    help="Select an image file (e.g., X-ray, CT, MRI)."
    )

if uploaded_file:
    image_bytes = uploaded_file.getvalue() # Read bytes first
    try:
        # Use io.BytesIO to open image from bytes
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception as e:
        st.error(f"Error opening image file: Seems it's not a valid image format.")
        st.exception(e)
        st.stop()

    st.image(image, caption="🖼️ Uploaded Image", use_container_width=True)
    st.markdown("---")
    st.markdown("#### Analysis Steps:")

    # Use st.status for better progress indication
    with st.status("1️⃣ Validating image as medical scan...", expanded=True) as status:
        is_valid_gemini = is_medical_scan_gemini(image_bytes)
        if is_valid_gemini:
            status.update(label="✅ Validation Successful (Likely Medical Scan)", state="complete", expanded=False)
        else:
             status.update(label="❌ Validation Failed (Not identified as Medical Scan)", state="error", expanded=True)
             st.info("Upload a clear X-ray, CT, MRI, or similar image.")
             st.stop() # Stop if not valid

    # Continue only if validation passed
    caption = "Error during captioning."
    with st.status("2️⃣ Generating Scan Description...", expanded=True) as status:
        try:
            inputs = caption_processor(image, return_tensors="pt")
            with torch.no_grad():
                out = caption_model.generate(**inputs, max_new_tokens=75, num_beams=5, early_stopping=True)
                caption = caption_processor.decode(out[0], skip_special_tokens=True)
            st.info(f"**Description:** {caption}")
            status.update(label="✅ Description Generated", state="complete", expanded=False)
        except Exception as e:
            st.exception(e)
            status.update(label="❌ Error Generating Description", state="error", expanded=True)


    suggestion = "Error during suggestion generation."
    with st.status("3️⃣ Generating Suggestion...", expanded=True) as status:
        try:
            # Refined prompt
            prompt = (
                f"Given the description of a medical image: '{caption}'. Suggest a broad, general next step "
                f"a medical professional might consider, or a general area of focus. Avoid specific diagnoses. Be concise."
            )
            text_inputs = text_tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
            with torch.no_grad():
                text_output = text_model.generate(
                    **text_inputs, max_new_tokens=100, num_beams=4, early_stopping=True, no_repeat_ngram_size=2
                )
                suggestion = text_tokenizer.decode(text_output[0], skip_special_tokens=True)
            st.warning(f"**Suggestion:** {suggestion}")
            status.update(label="✅ Suggestion Generated", state="complete", expanded=False)
        except Exception as e:
            st.exception(e)
            status.update(label="❌ Error Generating Suggestion", state="error", expanded=True)

    st.success("✨ Analysis Complete!")


st.markdown("---")
st.markdown("<div style='text-align: center;'>Developed with Hugging Face 🤗 & Google Gemini ✨</div>", unsafe_allow_html=True)