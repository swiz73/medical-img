import ollama
import streamlit as st
from google import genai
from PIL import Image
import tempfile

if 'image' not in st.session_state:
    st.session_state['image'] = None
if 'image_path' not in st.session_state:
    st.session_state['image_path'] = None

st.set_page_config(page_title="MedLens", page_icon="🔬", layout="wide")
st.title("🔬 Medical Image Analyzer")
st.markdown("Comparing **Gemini AI** and **MedGamma** on medical images")

st.sidebar.title("API Keys!")

model_choice = st.sidebar.radio(
    "Choose an AI model",
    ["Gemini", "Claude (Anthropic)"]
)

api_key = st.sidebar.text_input(
    "Enter Gemini API Key" if model_choice == "Gemini" else "Enter Anthropic API Key",
    type="password"
)

uploaded_file = st.file_uploader(
    "Upload a medical imageee",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.session_state['image'] = image

    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
        image.save(tmp.name)
        st.session_state['image_path'] = tmp.name

if st.session_state['image']:
    image = st.session_state['image']
    image_path = st.session_state['image_path']

    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("Image you provided")
        st.image(image, use_column_width=True)

    with col2:
        if model_choice == "Gemini":
            st.subheader("Gemini Analysis")
        else:
            st.subheader("Claude Analysis")

        if st.button("Analyze with AI"):
            with st.spinner("Analyzing..."):
                try:
                    if model_choice == "Gemini":
                        client = genai.Client(api_key=api_key)
                        response = client.models.generate_content(
                            model="gemini-2.0-flash",
                            contents=["""Analyze this medical image and provide:
1. Image Type (X-ray / MRI / CT Scan)
2. Body Part
3. Key Findings
4. Possible Diagnosis
5. Confidence Level (Low / Medium / High)
6. Recommended Next Steps""", image]
                        )
                        st.write(response.text)
                    else:
                        import anthropic, base64, io
                        buffer = io.BytesIO()
                        image.save(buffer, format="PNG")
                        img_base64 = base64.b64encode(buffer.getvalue()).decode()
                        client = anthropic.Anthropic(api_key=api_key)
                        response = client.messages.create(
                            model="claude-opus-4-6",
                            max_tokens=1024,
                            messages=[{
                                "role": "user",
                                "content": [
                                    {
                                        "type": "image",
                                        "source": {
                                            "type": "base64",
                                            "media_type": "image/png",
                                            "data": img_base64
                                        }
                                    },
                                    {
                                        "type": "text",
                                        "text": """Analyze this medical image and provide:
1. Image Type (X-ray / MRI / CT Scan)
2. Body Part
3. Key Findings
4. Possible Diagnosis
5. Confidence Level (Low / Medium / High)
6. Recommended Next Steps"""
                                    }
                                ]
                            }]
                        )
                        st.write(response.content[0].text)
                except Exception as e:
                    st.error(f"Error: {str(e)}")

    with col3:
        st.subheader("Gemma Analysis")
        if st.button("Analyze with Gemma"):
            with st.spinner("Gemma is thinking..."):
                try:
                    response = ollama.chat(
                        model="gemma3:4b",
                        messages=[{
                            "role": "user",
                            "content": """Analyze this medical image and provide:
1. Image Type (X-ray / MRI / CT Scan)
2. Body Part
3. Key Findings
4. Possible Diagnosis
5. Confidence Level (Low / Medium / High)
6. Recommended Next Steps""",
                            "images": [image_path]
                        }]
                    )
                    st.write(response['message']['content'])
                except Exception as e:
                    st.error(f"Error: {str(e)}")

else:
    st.info("Please upload a medical imageeee!!")