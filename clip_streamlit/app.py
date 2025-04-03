import streamlit as st
from PIL import Image
import torch
import os
import sys
import logging
import subprocess
import importlib
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def ensure_clip():
    """Ensure CLIP is installed from GitHub"""
    try:
        # Try to import existing CLIP installation
        try:
            clip_module = importlib.import_module('clip')
            
            # Check if it's the GitHub version
            if hasattr(clip_module, 'available_models'):
                logger.info("GitHub version of CLIP already installed")
                return clip_module
            
            # If not GitHub version, uninstall it
            logger.info("Non-GitHub CLIP version found. Uninstalling...")
            subprocess.check_call([
                sys.executable, "-m", "pip", "uninstall", "-y", "clip"
            ])
        except ImportError:
            logger.info("No existing CLIP installation found")
        
        # Install CLIP from GitHub
        logger.info("Installing CLIP from GitHub...")
        subprocess.check_call([
            sys.executable, "-m", "pip", "install",
            "git+https://github.com/openai/CLIP.git"
        ])
        
        # Clear importlib cache and import new installation
        importlib.invalidate_caches()
        clip_module = importlib.import_module('clip')
        
        # Verify the installation
        if hasattr(clip_module, 'available_models'):
            logger.info("CLIP installed successfully from GitHub")
            return clip_module
        else:
            raise ImportError("Invalid CLIP installation detected")
            
    except Exception as e:
        logger.error(f"Failed to install CLIP: {e}")
        raise

# Initialize CLIP model with better error handling
@st.cache_resource
def load_model():
    try:
        clip = ensure_clip()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {device}")
        
        model, preprocess = clip.load("ViT-B/32", device=device)
        model.eval()
        return model, preprocess, device
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        st.error(f"Failed to load CLIP model: {str(e)}")
        return None, None, "cpu"

def run_app():
    """Entry point for the application when used as a package."""
    try:
        # Ensure CLIP is installed before launching the app
        _ = ensure_clip()
        
        base_path = Path(__file__).parent / "base.py"
        
        # Use subprocess to run streamlit properly
        cmd = [sys.executable, "-m", "streamlit", "run", str(base_path)]
        subprocess.run(cmd)
        
    except Exception as e:
        logger.error(f"Failed to launch application: {e}")
        sys.exit(1)

if __name__ == "__main__":
    run_app()