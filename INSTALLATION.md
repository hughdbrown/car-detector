# Installation Guide

## Overview

This script uses a multi-stage AI approach for car detection:
1. **YOLO** for detecting vehicles in video frames
2. **PyTorch + EfficientNet** for classifying car make/model
3. **EasyOCR** for license plate recognition (with optional OpenALPR enhancement)

## Quick Start (uv - Recommended)

If you have [uv](https://github.com/astral-sh/uv) installed:

```bash
# All dependencies are auto-installed including EasyOCR
./car_detector.py video.mov
```

**That's it!** All required libraries including EasyOCR will be automatically installed.

## Traditional Installation

### Step 1: Install Python Dependencies

```bash
pip install -r requirements.txt
```

This installs:
- `opencv-python` - Video processing
- `ultralytics` - YOLO object detection
- `torch` + `torchvision` - PyTorch and EfficientNet
- `easyocr` - License plate OCR
- `click` - Command-line interface
- `Pillow` - Image processing
- `numpy` - Numerical operations

### Step 2: (Optional) Install OpenALPR for Enhanced Accuracy

**Note:** OpenALPR is **completely optional**. The script works out-of-the-box with EasyOCR.

OpenALPR can provide better license plate recognition in some cases, but it's not required.

#### ⚠️ macOS - OpenALPR Not Available via Homebrew

OpenALPR is **no longer available via Homebrew** (the homebrew-science formula was deprecated).

**Alternatives for macOS:**

1. **Use EasyOCR (default)** - Already included, no installation needed
2. **Build OpenALPR from source** - Complex, requires manual compilation
3. **Use commercial alternatives** - SimpleLPR or Plate Recognizer API

**To build from source on macOS (advanced):**
```bash
# Install dependencies via Homebrew
brew install opencv tesseract log4cplus

# Clone and build OpenALPR
git clone https://github.com/openalpr/openalpr.git
cd openalpr/src
mkdir build && cd build
cmake ..
make
sudo make install
```

See: https://github.com/openalpr/openalpr/issues/613

#### Ubuntu/Debian Linux

```bash
sudo apt-get update
sudo apt-get install -y openalpr openalpr-daemon openalpr-utils libopenalpr-dev
```

#### Windows

Download the installer from https://github.com/openalpr/openalpr/releases and add to PATH.

## Verifying Installation

### Test the Script

```bash
# Run on a sample video
./car_detector.py test_video.mov
```

You should see:
```
Loading YOLO model...
Loading EfficientNet model for car classification...
Loading EasyOCR for license plate recognition...
Using EasyOCR for license plate recognition
Processing video: test_video.mov
...
```

If OpenALPR is also installed, you'll see:
```
OpenALPR also detected - will use as fallback
```

### Verify OpenALPR (Optional)

If you installed OpenALPR:

```bash
alpr --version
```

Expected output:
```
openalpr version x.x.x
```

## Current Limitations

### Car Model Classification

The current implementation uses EfficientNet with **ImageNet weights** (not Stanford Cars dataset weights).

**What this means:**
- The classifier will return "Unknown Car Model" for all cars
- It won't distinguish between Toyota Sienna and Hyundai Ioniq 6

**To enable actual car model detection**, you need to:

1. **Download a pre-trained Stanford Cars model:**
   ```bash
   # Example: Download from a repository
   wget https://www.dropbox.com/s/w550z44ur2pwr4j/model_best.pth -O models/stanford_cars.pth
   ```

2. **Modify `CarModelClassifier.__init__()` in car_detector.py:**
   ```python
   # Replace line ~93:
   CarModelClassifier._model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)

   # With:
   CarModelClassifier._model = models.efficientnet_b0(num_classes=196)  # 196 Stanford Cars classes
   state_dict = torch.load('models/stanford_cars.pth', map_location=CarModelClassifier._device)
   CarModelClassifier._model.load_state_dict(state_dict)
   ```

3. **Add class mapping:**
   You'll need a mapping from the 196 Stanford Cars classes to determine which correspond to:
   - Toyota Sienna
   - Hyundai Ioniq 6

   See ANALYSIS.md for more details on training/integrating a custom model.

### License Plate Recognition

The script uses EasyOCR by default, which works well with:
- Clear, frontal or rear views of vehicles
- Good lighting conditions
- Minimal motion blur
- Most license plate formats (US, EU, etc.)

**Improvements:**
- Image preprocessing (adaptive thresholding) enhances contrast
- Pattern matching filters non-plate text
- Flexible character length (3-9 characters for different regions)
- Lower confidence threshold (0.3) for better detection

If OpenALPR is installed, it will automatically be used as a fallback when EasyOCR fails to detect a plate.

## Troubleshooting

### "alpr: command not found"

This is normal if you haven't installed OpenALPR. The script will work fine with EasyOCR alone.

If you want to install OpenALPR:
- On macOS: See the "Build from source" section above
- On Linux: `sudo apt-get install openalpr`
- Verify: `which alpr`

### "CUDA out of memory" errors

PyTorch is trying to use GPU but running out of memory.
- Force CPU mode by setting: `export CUDA_VISIBLE_DEVICES=-1`
- Or reduce batch size / video resolution

### Slow processing

- Ensure GPU acceleration is working: Check for "cuda" in startup messages
- Consider reducing video resolution
- Skip frames during processing (modify car_detector.py line ~289)

### Low car detection accuracy

- Try lowering the confidence threshold: `--confidence 0.3`
- Ensure adequate lighting in video
- Check that cars occupy sufficient frame area

## Performance Optimization

### GPU Acceleration

For much faster processing, ensure PyTorch can use your GPU:

```python
import torch
print(torch.cuda.is_available())  # Should print True
```

If False:
- Install CUDA-enabled PyTorch: https://pytorch.org/get-started/locally/
- Verify NVIDIA drivers are installed

### Frame Skipping

To process faster (with lower accuracy), modify `detect_cars()`:

```python
# Process every Nth frame instead of every frame
if frame_count % 5 != 0:  # Skip 4 out of 5 frames
    frame_count += 1
    continue
```

## Next Steps

1. For actual car model detection: See ANALYSIS.md for training instructions
2. For better license plate recognition: Consider commercial APIs like Plate Recognizer
3. For production use: Implement proper error handling and logging

## Resources

- OpenALPR: https://github.com/openalpr/openalpr
- Stanford Cars Dataset: https://datasets.activeloop.ai/docs/ml/datasets/stanford-cars-dataset/
- PyTorch EfficientNet Tutorial: https://debuggercafe.com/stanford-cars-classification-using-efficientnet-pytorch/
- YOLO Documentation: https://docs.ultralytics.com/
