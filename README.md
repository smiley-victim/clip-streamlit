# CLIP Streamlit

A Streamlit application for zero-shot image classification using OpenAI's CLIP model.

## Features

- Zero-shot image classification
- Interactive web interface
- Support for custom labels
- Confidence threshold adjustment
- Visual results with bar charts

## Installation

Simple installation:
```bash
pip install clip-streamlit
```

The package will automatically install CLIP when you first run the application.

## Usage

Run the application:
```bash
clip-streamlit
```

The first time you run it, it will install CLIP if needed.

Or use it in your Python code:

```python
from clip_streamlit import run_app

run_app()
```

## Requirements

- Python 3.7+
- Streamlit
- PyTorch
- CLIP
- Pillow

## License

MIT License