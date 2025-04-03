import streamlit as st
from PIL import Image
import torch
import clip
import os
import sys
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize CLIP model with better error handling
@st.cache_resource
def load_model():
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {device}")
        
        model, preprocess = clip.load("ViT-B/32", device=device)
        model.eval()
        return model, preprocess, device
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        st.error(f"Failed to load CLIP model: {str(e)}")
        return None, None, "cpu"

# Main app function
def main():
    st.set_page_config(page_title="Zero-Shot Image Detection", layout="wide")
    st.title("🔍 Zero-Shot Image Detection with CLIP")
    
    # Load model
    if 'model' not in st.session_state:
        model, preprocess, device = load_model()
        if model is None:
            st.error("Failed to initialize the model. Please refresh the page.")
            return
        st.session_state.model = model
        st.session_state.preprocess = preprocess
        st.session_state.device = device

    # UI Components
    with st.sidebar:
        st.header("Settings")
        uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])
        labels = st.text_area(
            "Enter labels (comma separated)", 
            "cat, dog, car, tree, person, building, landscape, food"
        )
        confidence_threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.5)
        temperature = st.slider("Temperature", 0.01, 1.0, 0.07)

    # Process labels
    labels = [label.strip() for label in labels.split(",") if label.strip()]

    if uploaded_file:
        try:
            # Display image and results in columns
            col1, col2 = st.columns([1, 2])
            
            with col1:
                image = Image.open(uploaded_file).convert('RGB')
                st.image(image, caption="Uploaded Image", use_container_width=True)
            
            with col2:
                if st.button("Classify Image"):
                    try:
                        # Prepare inputs
                        image_input = st.session_state.preprocess(image).unsqueeze(0)
                        image_input = image_input.to(st.session_state.device)
                        text_inputs = clip.tokenize(labels).to(st.session_state.device)

                        # Get predictions
                        with torch.no_grad():
                            image_features = st.session_state.model.encode_image(image_input)
                            text_features = st.session_state.model.encode_text(text_inputs)
                            
                            # Normalize features
                            image_features /= image_features.norm(dim=-1, keepdim=True)
                            text_features /= text_features.norm(dim=-1, keepdim=True)
                            
                            # Calculate similarity
                            similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
                            scores = similarity[0].cpu().numpy()

                        # Display results
                        results = sorted(zip(labels, scores), key=lambda x: x[1], reverse=True)
                        filtered_results = [(label, score) for label, score in results if score >= confidence_threshold]

                        if filtered_results:
                            st.write("### Results:")
                            for label, score in filtered_results:
                                st.write(f"**{label}**: {score:.2%}")
                            
                            # Create bar chart
                            chart_data = {
                                "Label": [r[0] for r in filtered_results],
                                "Confidence": [float(r[1]) for r in filtered_results]
                            }
                            st.bar_chart(data=chart_data)
                        else:
                            st.warning("No results above the confidence threshold.")
                            
                    except Exception as e:
                        logger.error(f"Classification error: {str(e)}")
                        st.error(f"Error during classification: {str(e)}")
        
        except Exception as e:
            logger.error(f"Error processing image: {str(e)}")
            st.error(f"Error processing image: {str(e)}")

if __name__ == "__main__":
    main()