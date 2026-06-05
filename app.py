import streamlit as st
from google import genai
from PIL import Image

st.set_page_config(page_title="MedLens", page_icon="🔬", layout="wide")
st.title("🔬 Medical Image Analyzer")
st.markdown("Analyzing medical images using **Gemini AI**")

api_key = st.sidebar.text_input("Enter Gemini API Key", type="password")

if api_key:
    client = genai.Client(api_key=api_key)

    uploaded_file = st.file_uploader(
        "Upload a medical image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("📷 Your Image")
            st.image(image, use_column_width=True)

        with col2:
            st.subheader("✦ Gemini Analysis")
            if st.button("Analyze"):
                with st.spinner("Analyzing..."):
                    try:
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
                    except Exception as e:
                        st.error(f"Error: {str(e)}")

else:
    st.warning("Please enter your Gemini API key in the sidebar")