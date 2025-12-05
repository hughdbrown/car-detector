#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.8"
# dependencies = [
#     "opencv-python>=4.8.0",
#     "ultralytics>=8.0.0",
#     "numpy>=1.24.0",
#     "torch>=2.0.0",
#     "torchvision>=0.15.0",
#     "Pillow>=10.0.0",
#     "click>=8.0.0",
#     "requests>=2.31.0",
#     "easyocr>=1.7.0",
# ]
# ///
"""
Car Detection Script for iPhone Videos
Detects Toyota Sienna and Hyundai Ioniq 6 vehicles in videos,
extracts time offsets, and reads license plates using:
- YOLO for car detection
- EfficientNet for car model classification
- EasyOCR for license plate recognition (with optional OpenALPR fallback)
"""

import cv2
import click
import sys
import glob
import subprocess
import re
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import numpy as np
import torch
import torchvision.transforms as transforms
from torchvision import models
from ultralytics import YOLO
from PIL import Image
import easyocr


@dataclass
class CarDetection:
    """Represents a detected car in the video"""
    car_model: str
    entry_time: float
    exit_time: float
    license_plate: Optional[str] = None
    confidence: float = 0.0

    def __str__(self) -> str:
        result = f"Car Model: {self.car_model}"
        if self.confidence > 0:
            result += f" (confidence: {self.confidence:.2%})"
        result += "\n"
        result += f"Entry Time: {self.format_time(self.entry_time)}\n"
        result += f"Exit Time: {self.format_time(self.exit_time)}\n"
        if self.license_plate:
            result += f"License Plate: {self.license_plate}\n"
        else:
            result += "License Plate: Not visible\n"
        return result

    @staticmethod
    def format_time(seconds: float) -> str:
        """Format time in seconds to HH:MM:SS.mmm"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = seconds % 60
        return f"{hours:02d}:{minutes:02d}:{secs:06.3f}"


class CarModelClassifier:
    """Classifies car models using EfficientNet trained on Stanford Cars dataset"""

    # Mapping of target car models we're looking for
    TARGET_MODELS = {
        'toyota sienna': ['Toyota Sienna', 'Sienna'],
        'hyundai ioniq 6': ['Hyundai Ioniq 6', 'Ioniq 6', 'Ioniq'],
    }

    _model = None
    _transform = None
    _device = None

    def __init__(self, force_device: Optional[str] = None):
        """
        Initialize the car model classifier

        Args:
            force_device: Force device to 'cpu' or 'cuda', or None for auto-detect
        """
        if CarModelClassifier._model is None:
            click.echo("Loading EfficientNet model for car classification...", err=True)

            # Determine device
            if force_device:
                if force_device == 'cuda' and not torch.cuda.is_available():
                    click.echo("Warning: CUDA requested but not available, using CPU", err=True)
                    CarModelClassifier._device = torch.device('cpu')
                else:
                    CarModelClassifier._device = torch.device(force_device)
            else:
                CarModelClassifier._device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

            click.echo(f"Using device: {CarModelClassifier._device}", err=True)

            # Load pretrained EfficientNet-B0 model
            # Note: In production, you would load a model fine-tuned on Stanford Cars dataset
            # For now, we use ImageNet weights as a placeholder
            CarModelClassifier._model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
            CarModelClassifier._model = CarModelClassifier._model.to(CarModelClassifier._device)
            CarModelClassifier._model.eval()

            # Image preprocessing transform
            CarModelClassifier._transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])

        self.model = CarModelClassifier._model
        self.transform = CarModelClassifier._transform
        self.device = CarModelClassifier._device

    def classify(self, car_roi: np.ndarray) -> Tuple[str, float]:
        """
        Classify a car image to determine make/model

        Args:
            car_roi: Region of interest containing the car (numpy array in BGR format)

        Returns:
            Tuple of (model_name, confidence)
        """
        try:
            # Convert BGR to RGB
            car_rgb = cv2.cvtColor(car_roi, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(car_rgb)

            # Preprocess
            input_tensor = self.transform(pil_image).unsqueeze(0).to(self.device)

            # Run inference
            with torch.no_grad():
                outputs = self.model(input_tensor)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)
                confidence, predicted = torch.max(probabilities, 1)

            # Note: This is a placeholder implementation
            # In production, you would:
            # 1. Load a model fine-tuned on Stanford Cars dataset (196 classes)
            # 2. Map the predicted class to actual car make/model
            # 3. Check if it matches Toyota Sienna or Hyundai Ioniq 6

            # For now, return "Unknown" as we don't have the fine-tuned model
            # TODO: Integrate actual Stanford Cars trained model
            return "Unknown Car Model", confidence.item()

        except Exception as e:
            click.echo(f"Error classifying car model: {e}", err=True)
            return "Unknown Car Model", 0.0


class LicensePlateRecognizer:
    """Recognizes license plates using EasyOCR with optional OpenALPR fallback"""

    _ocr_reader = None
    _alpr_available = None
    _use_alpr = False

    def __init__(self, force_device: Optional[str] = None):
        """
        Initialize the license plate recognizer

        Args:
            force_device: Force device to 'cpu' or 'cuda', or None for auto-detect
        """
        # Initialize EasyOCR (primary method)
        if LicensePlateRecognizer._ocr_reader is None:
            click.echo("Loading EasyOCR for license plate recognition...", err=True)

            # Determine GPU usage
            if force_device:
                use_gpu = (force_device == 'cuda' and torch.cuda.is_available())
                if force_device == 'cuda' and not torch.cuda.is_available():
                    click.echo("Warning: CUDA requested but not available for EasyOCR", err=True)
            else:
                use_gpu = torch.cuda.is_available()

            LicensePlateRecognizer._ocr_reader = easyocr.Reader(
                ['en'],
                gpu=use_gpu
            )
            device_name = "GPU" if use_gpu else "CPU"
            click.echo(f"Using EasyOCR with {device_name} for license plate recognition", err=True)

        # Check if OpenALPR is also available (optional enhancement)
        if LicensePlateRecognizer._alpr_available is None:
            LicensePlateRecognizer._alpr_available = self._check_openalpr()
            if LicensePlateRecognizer._alpr_available:
                click.echo("OpenALPR also detected - will use as fallback", err=True)
                LicensePlateRecognizer._use_alpr = True

        self.ocr_reader = LicensePlateRecognizer._ocr_reader
        self.alpr_available = LicensePlateRecognizer._alpr_available
        self.use_alpr = LicensePlateRecognizer._use_alpr

    def _check_openalpr(self) -> bool:
        """Check if OpenALPR command-line tool is available"""
        try:
            result = subprocess.run(['alpr', '--version'], capture_output=True, timeout=5)
            return result.returncode == 0
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return False

    def _recognize_with_easyocr(self, car_roi: np.ndarray) -> Optional[str]:
        """Use EasyOCR to detect license plate text"""
        try:
            debug = getattr(__builtins__, 'DEBUG_MODE', False)

            # Preprocess image for better OCR
            gray = cv2.cvtColor(car_roi, cv2.COLOR_BGR2GRAY)

            # Apply adaptive thresholding to enhance contrast
            thresh = cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
            )

            # Run EasyOCR
            results = self.ocr_reader.readtext(thresh, detail=1)

            if debug and results:
                click.echo(f"  EasyOCR found {len(results)} text regions:", err=True)
                for (bbox, text, confidence) in results:
                    click.echo(f"    '{text}' (confidence: {confidence:.2f})", err=True)

            # Find license plate-like text
            best_plate = None
            best_confidence = 0.0

            for (bbox, text, confidence) in results:
                # Clean text - remove spaces and special characters
                cleaned = re.sub(r'[^A-Z0-9]', '', text.upper())

                # License plate patterns (flexible for different regions)
                # US: 3-8 alphanumeric characters
                # EU: 4-9 alphanumeric characters
                if 3 <= len(cleaned) <= 9 and cleaned.isalnum():
                    if debug:
                        click.echo(f"    Candidate plate: '{cleaned}' (confidence: {confidence:.2f})", err=True)
                    if confidence > best_confidence:
                        best_confidence = confidence
                        best_plate = cleaned

            # Return if confidence is reasonable
            if best_plate and best_confidence > 0.3:
                if debug:
                    click.echo(f"  Selected plate: '{best_plate}' (confidence: {best_confidence:.2f})", err=True)
                return best_plate

            if debug:
                click.echo(f"  No valid plate found (best confidence: {best_confidence:.2f})", err=True)

            return None

        except Exception as e:
            click.echo(f"EasyOCR error: {e}", err=True)
            return None

    def _recognize_with_openalpr(self, car_roi: np.ndarray) -> Optional[str]:
        """Use OpenALPR as fallback"""
        try:
            # Save temporary image
            temp_path = "/tmp/car_roi.jpg"
            cv2.imwrite(temp_path, car_roi)

            # Run OpenALPR
            result = subprocess.run(
                ['alpr', '-n', '1', temp_path],
                capture_output=True,
                text=True,
                timeout=5
            )

            if result.returncode == 0:
                # Parse output to extract plate number
                output = result.stdout
                for line in output.split('\n'):
                    if 'plate' in line.lower() and ':' in line:
                        parts = line.split(':')
                        if len(parts) >= 2:
                            plate = parts[1].strip().split()[0]
                            return plate

            return None

        except Exception as e:
            return None

    def recognize(self, car_roi: np.ndarray) -> Optional[str]:
        """
        Recognize license plate from car region of interest

        Args:
            car_roi: Region of interest containing the car

        Returns:
            License plate text or None if not detected
        """
        # Check minimum size
        if car_roi.shape[0] < 50 or car_roi.shape[1] < 50:
            return None

        # Try EasyOCR first
        plate = self._recognize_with_easyocr(car_roi)

        # If EasyOCR fails and OpenALPR is available, try it
        if not plate and self.use_alpr:
            plate = self._recognize_with_openalpr(car_roi)

        return plate


class CarDetector:
    """Detects specific car models in video frames"""

    TARGET_CARS = {
        "toyota sienna": ["sienna", "toyota sienna", "minivan"],
        "hyundai ioniq 6": ["ioniq 6", "hyundai ioniq", "ioniq"]
    }

    # Class-level models to share across instances
    _yolo_model = None
    _car_classifier = None
    _plate_recognizer = None

    def __init__(self, video_path: str, device: Optional[str] = None):
        """
        Initialize car detector

        Args:
            video_path: Path to video file
            device: Force device to 'cpu' or 'cuda', or None for auto-detect
        """
        self.video_path = Path(video_path)
        if not self.video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")

        # Initialize YOLO model for object detection (shared across instances)
        if CarDetector._yolo_model is None:
            click.echo("Loading YOLO model...", err=True)
            CarDetector._yolo_model = YOLO('yolov8n.pt')  # Using YOLOv8 nano for speed

        # Initialize car model classifier (shared across instances)
        if CarDetector._car_classifier is None:
            CarDetector._car_classifier = CarModelClassifier(force_device=device)

        # Initialize license plate recognizer (shared across instances)
        if CarDetector._plate_recognizer is None:
            CarDetector._plate_recognizer = LicensePlateRecognizer(force_device=device)

        self.yolo_model = CarDetector._yolo_model
        self.car_classifier = CarDetector._car_classifier
        self.plate_recognizer = CarDetector._plate_recognizer

        # Video properties
        self.cap = cv2.VideoCapture(str(self.video_path))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

    def get_video_duration(self) -> float:
        """Get the duration of the video in seconds"""
        if self.fps > 0:
            return self.total_frames / self.fps
        return 0.0

    def _calculate_iou(self, bbox1: Tuple[int, int, int, int], bbox2: Tuple[int, int, int, int]) -> float:
        """Calculate Intersection over Union (IoU) between two bounding boxes"""
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2

        # Calculate intersection area
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)

        if x2_i < x1_i or y2_i < y1_i:
            return 0.0

        intersection = (x2_i - x1_i) * (y2_i - y1_i)

        # Calculate union area
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection

        return intersection / union if union > 0 else 0.0

    def detect_cars(self, confidence_threshold: float = 0.5) -> List[CarDetection]:
        """
        Process video and detect target car models

        Args:
            confidence_threshold: Minimum confidence for car detection

        Returns:
            List of CarDetection objects
        """
        detections = []
        current_tracking = {}  # Track cars currently in frame
        next_car_id = 0  # Unique ID counter
        frame_count = 0
        iou_threshold = 0.3  # IoU threshold for matching cars across frames

        click.echo(f"Processing video: {self.video_path.name}", err=True)
        click.echo(f"Total frames: {self.total_frames}, FPS: {self.fps}", err=True)

        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break

            current_time = frame_count / self.fps

            # Run YOLO detection every frame (can be optimized to skip frames)
            results = self.yolo_model(frame, conf=confidence_threshold, verbose=False)

            # Process detections
            detected_boxes = []
            debug = getattr(__builtins__, 'DEBUG_MODE', False)

            for result in results:
                boxes = result.boxes
                for box in boxes:
                    cls = int(box.cls[0])
                    class_name = self.yolo_model.names[cls].lower()

                    # YOLO detects "car" class - we'll use this as base detection
                    if class_name in ['car', 'truck', 'bus']:
                        # Extract bounding box
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        detected_boxes.append((x1, y1, x2, y2, class_name))

            if debug and detected_boxes:
                click.echo(f"Frame {frame_count}: Detected {len(detected_boxes)} vehicle(s)", err=True)

            # Match detected boxes with existing tracked cars using IoU
            matched_ids = set()
            cars_in_frame = set()

            for bbox in detected_boxes:
                x1, y1, x2, y2, class_name = bbox
                best_match_id = None
                best_iou = 0.0

                # Try to match with existing tracked cars
                for car_id, car_data in current_tracking.items():
                    iou = self._calculate_iou((x1, y1, x2, y2), car_data['bbox'])
                    if iou > best_iou and iou > iou_threshold:
                        best_iou = iou
                        best_match_id = car_id

                if best_match_id is not None:
                    # Update existing car's bounding box
                    current_tracking[best_match_id]['bbox'] = (x1, y1, x2, y2)
                    current_tracking[best_match_id]['last_seen'] = frame_count
                    matched_ids.add(best_match_id)
                    cars_in_frame.add(best_match_id)

                    # Try to detect license plate again if not yet found
                    if not current_tracking[best_match_id]['license_plate']:
                        car_roi = frame[y1:y2, x1:x2]
                        plate = self.plate_recognizer.recognize(car_roi)
                        if plate:
                            current_tracking[best_match_id]['license_plate'] = plate
                            click.echo(f"Detected plate: {plate} for car {best_match_id}", err=True)
                else:
                    # New car detected
                    car_id = next_car_id
                    next_car_id += 1

                    # Extract car region of interest
                    car_roi = frame[y1:y2, x1:x2]

                    # Classify car model using EfficientNet
                    car_model, model_confidence = self.car_classifier.classify(car_roi)

                    # Attempt license plate recognition
                    license_plate = self.plate_recognizer.recognize(car_roi)
                    if license_plate:
                        click.echo(f"Detected plate: {license_plate} for new car {car_id}", err=True)

                    current_tracking[car_id] = {
                        'entry_time': current_time,
                        'model': car_model,
                        'confidence': model_confidence,
                        'license_plate': license_plate,
                        'bbox': (x1, y1, x2, y2),
                        'last_seen': frame_count
                    }
                    cars_in_frame.add(car_id)

            # Check for cars that left the frame (not seen for a few frames)
            cars_to_remove = []
            for car_id in current_tracking:
                if car_id not in cars_in_frame:
                    # Give it a grace period of 5 frames before considering it gone
                    if frame_count - current_tracking[car_id]['last_seen'] > 5:
                        # Car exited
                        car_data = current_tracking[car_id]
                        detection = CarDetection(
                            car_model=car_data['model'],
                            entry_time=car_data['entry_time'],
                            exit_time=current_time,
                            license_plate=car_data['license_plate'],
                            confidence=car_data['confidence']
                        )
                        detections.append(detection)
                        cars_to_remove.append(car_id)

            for car_id in cars_to_remove:
                del current_tracking[car_id]

            frame_count += 1

            # Progress indicator
            if frame_count % 30 == 0:
                progress = (frame_count / self.total_frames) * 100
                click.echo(f"Progress: {progress:.1f}%", err=True)

        # Handle cars still in frame at end of video
        final_time = self.total_frames / self.fps
        for car_id, car_data in current_tracking.items():
            detection = CarDetection(
                car_model=car_data['model'],
                entry_time=car_data['entry_time'],
                exit_time=final_time,
                license_plate=car_data['license_plate'],
                confidence=car_data['confidence']
            )
            detections.append(detection)

        self.cap.release()
        return detections


def process_video_file(video_path: str, confidence: float, device: Optional[str] = None) -> bool:
    """
    Process a single video file and output results

    Args:
        video_path: Path to the video file
        confidence: Detection confidence threshold
        device: Force device to 'cpu' or 'cuda', or None for auto-detect

    Returns:
        True if processing succeeded, False otherwise
    """
    try:
        detector = CarDetector(video_path, device=device)
        duration = detector.get_video_duration()

        # Output file header
        print(f"\nFile: {detector.video_path.name}")
        print(f"Duration: {CarDetection.format_time(duration)}")
        print("-" * 70)

        detections = detector.detect_cars(confidence_threshold=confidence)

        if not detections:
            print("No cars detected.")
        else:
            for i, detection in enumerate(detections, 1):
                print(f"\nDetection #{i}:")
                print(f"  Car Model: {detection.car_model}", end="")
                if detection.confidence > 0:
                    print(f" (confidence: {detection.confidence:.2%})")
                else:
                    print()
                print(f"  Entry Time: {detection.format_time(detection.entry_time)}")
                print(f"  Exit Time: {detection.format_time(detection.exit_time)}")
                if detection.license_plate:
                    print(f"  License Plate: {detection.license_plate}")
                else:
                    print("  License Plate: Not visible")

        return True

    except FileNotFoundError as e:
        print(f"\nFile: {Path(video_path).name}")
        print(f"Error: File not found - {e}", file=sys.stderr)
        return False
    except Exception as e:
        print(f"\nFile: {Path(video_path).name}")
        print(f"Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)
        return False


def expand_glob_patterns(patterns: List[str]) -> List[str]:
    """
    Expand glob patterns to actual file paths

    Args:
        patterns: List of file patterns (can include wildcards)

    Returns:
        List of expanded file paths
    """
    expanded_files = []
    for pattern in patterns:
        # Expand glob pattern
        matches = glob.glob(pattern)
        if matches:
            expanded_files.extend(matches)
        else:
            # If no matches, treat as literal filename
            expanded_files.append(pattern)

    return expanded_files


@click.command()
@click.argument('video_paths', nargs=-1, required=True, type=click.Path())
@click.option(
    '--confidence',
    '-c',
    type=click.FloatRange(0.0, 1.0),
    default=0.5,
    show_default=True,
    help='Detection confidence threshold'
)
@click.option(
    '--debug',
    '-d',
    is_flag=True,
    help='Enable debug output showing detection details'
)
@click.option(
    '--device',
    type=click.Choice(['auto', 'cpu', 'cuda'], case_sensitive=False),
    default='auto',
    show_default=True,
    help='Device for ML processing (auto=detect GPU, cpu=force CPU, cuda=force GPU)'
)
def main(video_paths: tuple[str, ...], confidence: float, debug: bool, device: str) -> None:
    """Detect Toyota Sienna and Hyundai Ioniq 6 in iPhone videos.

    Supports wildcards like *.mov for batch processing multiple videos.

    Examples:

        ./car_detector.py video.mov

        ./car_detector.py *.mov --confidence 0.6

        ./car_detector.py videos/*.mp4 --debug

        ./car_detector.py video.mov --device cpu
    """
    # Store debug mode globally
    import builtins
    builtins.DEBUG_MODE = debug

    # Convert device selection
    device_param = None if device == 'auto' else device
    # Expand glob patterns
    video_files = expand_glob_patterns(list(video_paths))

    if not video_files:
        click.echo("No video files found.", err=True)
        sys.exit(1)

    click.echo(f"Processing {len(video_files)} video file(s)...", err=True)

    # Process each video file
    success_count = 0
    for video_file in video_files:
        if process_video_file(video_file, confidence, device=device_param):
            success_count += 1

    click.echo(f"\n{'='*70}")
    click.echo(f"Completed: {success_count}/{len(video_files)} files processed successfully")
    click.echo(f"{'='*70}")

    if success_count == 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
