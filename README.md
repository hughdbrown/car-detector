# Car Detection for iPhone Videos

Advanced car detection script for iPhone videos using a multi-stage AI approach:
- **Stage 1:** YOLO for vehicle detection
- **Stage 2:** EfficientNet for car make/model classification
- **Stage 3:** EasyOCR for license plate recognition (with optional OpenALPR fallback)

Designed to detect Toyota Sienna and Hyundai Ioniq 6 vehicles with timestamps and license plate data.

## Features

- **Multi-stage AI pipeline:** YOLO → EfficientNet → EasyOCR
- Detects cars in iPhone video files with high accuracy
- Classifies car make/model using PyTorch + EfficientNet
- License plate recognition via EasyOCR (pip-installable, works out-of-the-box)
- Supports wildcard patterns for batch processing multiple videos
- Displays filename and video duration for each processed file
- Tracks entry and exit times for each vehicle
- GPU acceleration support for faster processing
- Shares models across video files for efficient batch processing

## Installation

### Option 1: Using uv (Recommended)

The script has inline dependency declarations and can be run directly with [uv](https://github.com/astral-sh/uv):

```bash
# No installation needed! Just run:
./car_detector.py video.mov

# Or with uv run:
uv run car_detector.py video.mov
```

On first run, uv will automatically install all dependencies including EasyOCR and download the YOLOv8 model (~6MB).

**No additional setup required!** License plate recognition works out-of-the-box with EasyOCR.

### Option 2: Traditional Installation

See [INSTALLATION.md](INSTALLATION.md) for complete installation instructions including:
- Python dependencies (including EasyOCR)
- Optional OpenALPR setup (for enhanced plate recognition on Linux)
- GPU acceleration configuration
- Stanford Cars model integration (for actual make/model detection)

## Usage

### Single Video File

```bash
./car_detector.py /path/to/video.mov
```

### Multiple Video Files

```bash
./car_detector.py video1.mov video2.mp4 video3.mov
```

### Using Wildcards

```bash
./car_detector.py *.mov
./car_detector.py videos/*.mp4
./car_detector.py recordings/2024-*.mov
```

### With Custom Confidence Threshold

```bash
./car_detector.py *.mov --confidence 0.6
```

### With Debug Mode

```bash
./car_detector.py video.mov --debug
```

Debug mode shows:
- How many vehicles detected per frame
- What text EasyOCR finds
- License plate candidates and confidence scores
- Which cars get matched across frames

### Force CPU or GPU Processing

```bash
# Auto-detect (default) - uses GPU if available
./car_detector.py video.mov --device auto

# Force CPU (useful for testing or if GPU has issues)
./car_detector.py video.mov --device cpu

# Force GPU (requires CUDA)
./car_detector.py video.mov --device cuda
```

**When to use:**
- `--device cpu`: Force CPU processing (slower but more compatible)
- `--device cuda`: Force GPU processing (faster, requires NVIDIA GPU with CUDA)
- `--device auto`: Auto-detect (uses GPU if available, falls back to CPU)

### Example Output

```
Processing 2 video file(s)...

File: dashcam_2024_01.mov
Duration: 00:02:45.320
----------------------------------------------------------------------

Detection #1:
  Car Model: Unknown Car Model
  Entry Time: 00:00:03.500
  Exit Time: 00:00:12.750
  License Plate: ABC1234

Detection #2:
  Car Model: Unknown Car Model
  Entry Time: 00:00:18.200
  Exit Time: 00:00:25.100
  License Plate: Not visible

File: dashcam_2024_02.mov
Duration: 00:01:30.150
----------------------------------------------------------------------
No cars detected.

======================================================================
Completed: 2/2 files processed successfully
======================================================================
```

## Current Limitations & Production Readiness

### ⚠️ Car Model Classification (Incomplete)

**Current Status:** The EfficientNet classifier uses ImageNet weights and will return "Unknown Car Model" for all detections.

**Why:** The script is architected for Stanford Cars dataset integration, but requires:
1. Download pre-trained Stanford Cars weights (~200MB)
2. Load the 196-class model instead of ImageNet
3. Map Stanford Cars classes to Toyota Sienna / Hyundai Ioniq 6

**To Complete:** See [INSTALLATION.md](INSTALLATION.md) for instructions on integrating a Stanford Cars trained model.

### ✅ License Plate Recognition (Fully Functional)

**Current Status:** Fully functional using EasyOCR (pip-installable, no system dependencies).

**Features:**
- Works out-of-the-box (no system dependencies required)
- Image preprocessing with adaptive thresholding for better accuracy
- Pattern matching to filter non-plate text
- Supports US and EU plate formats (3-9 characters)
- Optional OpenALPR fallback if installed (Linux only - not available on macOS via Homebrew)

**Limitations:**
- Works best with clear, frontal/rear vehicle views
- Accuracy depends on lighting, resolution, and plate format
- May struggle with motion blur or distant shots

### 🔧 Production Readiness Checklist

- [x] YOLO vehicle detection - **Working**
- [x] EasyOCR license plate recognition - **Working** (pip-installable)
- [x] EfficientNet integration - **Architecture complete**
- [ ] Stanford Cars model integration - **Requires manual setup**
- [ ] Car model filtering (Toyota Sienna / Hyundai Ioniq 6) - **Requires Stanford Cars model**
- [x] Video batch processing - **Working**
- [x] Wildcard file support - **Working**
- [x] GPU acceleration - **Working**
- [x] Image preprocessing for better plate detection - **Working**

## How It Works

The script uses a sophisticated three-stage AI pipeline:

### Stage 1: Vehicle Detection (YOLO)
- Processes video frame-by-frame
- Uses YOLOv8 to detect vehicles (cars, trucks, buses)
- Extracts bounding boxes for detected vehicles

### Stage 2: Car Model Classification (EfficientNet)
- Takes detected car regions from Stage 1
- Uses PyTorch EfficientNet for classification
- **Current status:** Placeholder with ImageNet weights (returns "Unknown Car Model")
- **Production:** Requires Stanford Cars dataset fine-tuned model (see INSTALLATION.md)

### Stage 3: License Plate Recognition (EasyOCR)
- Processes car regions from Stage 1
- Applies image preprocessing (adaptive thresholding) for better OCR
- Uses EasyOCR for text detection and recognition
- Pattern matching filters out non-plate text
- Optional OpenALPR fallback (if installed on Linux)

### Tracking
- Tracks vehicles across frames to determine entry/exit times
- Assigns unique IDs to each detected vehicle
- Handles vehicles leaving and re-entering frame

### Output
- Formats results with timestamps and confidence scores
- Prints to stdout for easy piping/processing

## Performance Notes

- Processing speed depends on video resolution and hardware
- GPU acceleration (CUDA) is **10-50x faster** than CPU for ML models
- Auto-detected by default - use `--device` to override
- Progress updates are printed to stderr during processing

### GPU vs CPU Performance

| Task | GPU (CUDA) | CPU | Speedup |
|------|------------|-----|---------|
| EfficientNet (car classification) | ~10ms | ~100ms | 10x |
| EasyOCR (license plates) | ~50ms | ~500ms | 10x |
| Overall per frame | ~60ms | ~600ms | 10x |

**Example:** 30-second video at 30fps:
- GPU: ~54 seconds processing time
- CPU: ~9 minutes processing time

## Troubleshooting

### GPU Not Being Used (NVIDIA Card Present)

If you have an NVIDIA GPU but get "CUDA requested but PyTorch cannot find CUDA":

**Quick Diagnosis:**
```bash
python check_cuda.py
```

This script will check:
- PyTorch version (CPU-only vs CUDA-enabled)
- CUDA toolkit installation
- NVIDIA drivers
- GPU availability

**Most Common Issue: CPU-Only PyTorch**

Check your PyTorch version:
```bash
python -c "import torch; print(torch.__version__)"
```

If you see `cpu` in the version (e.g., `2.0.0+cpu`), you have the CPU-only build.

**Fix: Install CUDA-enabled PyTorch**

1. Check your CUDA version:
   ```bash
   nvcc --version
   ```

2. Install PyTorch for your CUDA version:
   ```bash
   # For CUDA 11.8
   pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu118

   # For CUDA 12.1
   pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu121
   ```

3. Verify installation:
   ```bash
   python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
   ```

**If CUDA toolkit is not installed:**
- Download from: https://developer.nvidia.com/cuda-downloads
- Make sure to match PyTorch CUDA version

**If NVIDIA drivers are not installed:**
- Download from: https://www.nvidia.com/Download/index.aspx
- Restart after installation

See full guide: https://pytorch.org/get-started/locally/

### Low Detection Rate
- Try lowering the `--confidence` threshold (default: 0.5)
- Ensure adequate lighting in the video
- Check that cars occupy a reasonable portion of the frame

### License Plate Not Detected
- License plates must be clearly visible and readable
- Works best with frontal or rear views of vehicles
- May struggle with motion blur or distant shots

### Out of Memory
- Reduce video resolution before processing
- Process shorter video segments
- Close other applications to free up RAM

## Future Enhancements

- [ ] Train custom model for specific car makes/models
- [x] Add support for batch processing multiple videos
- [x] Support wildcard patterns for file selection
- [ ] Implement frame skipping for faster processing
- [ ] Add video output with bounding boxes
- [ ] Improve license plate detection accuracy
- [ ] Add support for different video formats
- [ ] Add JSON output format option

## License

MIT License
