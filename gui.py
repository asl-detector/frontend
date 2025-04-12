import sys
import os
import json
import cv2
import pandas as pd
import subprocess
import pickle
from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QLabel,
    QFileDialog,
    QProgressBar,
    QSlider,
    QListWidget,
    QSplitter,
    QScrollArea,
    QMessageBox,
    QShortcut,
    QFrame,
    QInputDialog,
    QSizePolicy,
    QDialog,
    QTextEdit,
    QDialogButtonBox,
)
from PyQt5.QtCore import QThread, pyqtSignal, Qt, QTimer, QRect
from PyQt5.QtGui import QImage, QPixmap, QPainter, QColor, QPen, QKeySequence

# Import from existing backend
from extract import extract_features


def run_pose_estimation(video_path, output_path, model_path):
    """Run the pose estimation script (pe_cli.py) on the input video"""
    print(f"Running pose estimation on {video_path}...")

    # Build the command
    cmd = [
        "python",
        "pe_cli.py",
        "--video",
        video_path,
        "--output",
        output_path,
        "--model",
        model_path,
    ]

    print(f"Running command: {' '.join(cmd)}")

    # Run the command
    try:
        subprocess.run(cmd, check=True)

        # Check if the output file exists
        if os.path.exists(output_path):
            return output_path
        else:
            print(f"Warning: Expected output file {output_path} not found")
            return None
    except subprocess.CalledProcessError as e:
        print(f"Error running pose estimation: {e}")
        return None


def load_model(model_path):
    """Load the trained ASL detection model"""
    print(f"Loading model from {model_path}...")
    try:
        with open(model_path, "rb") as f:
            model_data = pickle.load(f)
        return model_data
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)


class AnnotationDialog(QDialog):
    """Custom dialog for entering annotation text with a larger text box"""

    def __init__(self, parent=None, default_text=""):
        super().__init__(parent)
        self.setWindowTitle("ASL Translation")
        self.setMinimumWidth(400)
        self.setMinimumHeight(200)

        layout = QVBoxLayout(self)

        # Add label
        label = QLabel("Enter the ASL translation:")
        layout.addWidget(label)

        # Add text edit with larger size
        self.text_edit = QTextEdit()
        self.text_edit.setMinimumHeight(100)
        self.text_edit.setText(default_text)
        layout.addWidget(self.text_edit)

        # Add buttons
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def get_text(self):
        """Return the entered text"""
        return self.text_edit.toPlainText()


class BatchProcessingThread(QThread):
    """Thread for processing multiple videos in a folder"""

    progress_update = pyqtSignal(int, int)  # current, total
    video_processed = pyqtSignal(str, bool)  # path, contains_asl
    processing_complete = pyqtSignal(list)  # list of ASL video paths
    processing_error = pyqtSignal(str, str)  # video_path, error_message

    def __init__(self, folder_path, model_path, mediapipe_model_path, output_dir):
        super().__init__()
        self.folder_path = folder_path
        self.model_path = model_path
        self.mediapipe_model_path = mediapipe_model_path
        self.output_dir = output_dir
        self.video_extensions = [".mp4", ".avi", ".mov", ".mkv", ".MOV"]  # Added .MOV

    def run(self):
        try:
            # Load ASL detection model
            model_data = load_model(self.model_path)

            # Find all videos in the folder
            video_files = []
            for file in os.listdir(self.folder_path):
                ext = os.path.splitext(file)[1].lower()
                if ext.lower() in [x.lower() for x in self.video_extensions]:
                    video_files.append(os.path.join(self.folder_path, file))

            asl_videos = []
            already_processed_count = 0

            # Process each video
            for i, video_path in enumerate(video_files):
                try:
                    self.progress_update.emit(i, len(video_files))

                    # Create output paths
                    video_name = os.path.splitext(os.path.basename(video_path))[0]
                    landmark_json_path = os.path.join(
                        self.output_dir, f"{video_name}_landmarks.json"
                    )
                    result_path = os.path.join(
                        self.output_dir, f"{video_name}_asl_detection.json"
                    )

                    # Check if this video has already been processed
                    if os.path.exists(result_path) and os.path.exists(
                        landmark_json_path
                    ):
                        # Load existing results
                        try:
                            with open(result_path, "r") as f:
                                result = json.load(f)
                                contains_asl = result.get("contains_asl", False)

                                # Emit signal for already processed video
                                self.video_processed.emit(video_path, contains_asl)

                                # Add to ASL videos list if contains ASL
                                if contains_asl:
                                    asl_videos.append(video_path)

                                already_processed_count += 1
                                continue  # Skip to next video
                        except Exception as e:
                            print(
                                f"Error loading existing result for {video_path}: {e}"
                            )
                            # If error loading results, process the video again

                    # Run pose estimation
                    landmark_file = run_pose_estimation(
                        video_path, landmark_json_path, self.mediapipe_model_path
                    )

                    if landmark_file:
                        # Process landmarks
                        contains_asl, avg_confidence = self.process_landmarks(
                            landmark_file, model_data
                        )

                        print(
                            f"Video {video_path}: contains_asl={contains_asl}, confidence={avg_confidence:.2f}"
                        )

                        result = {
                            "video": str(video_path),
                            "landmark_file": str(landmark_file),
                            "contains_asl": bool(contains_asl),
                            "average_confidence": float(avg_confidence),
                        }

                        with open(result_path, "w") as f:
                            json.dump(result, f, indent=2)

                        # Emit progress signal
                        self.video_processed.emit(video_path, bool(contains_asl))

                        # Add to ASL videos list if contains ASL
                        if contains_asl:
                            asl_videos.append(video_path)

                except Exception as e:
                    import traceback

                    traceback.print_exc()
                    self.processing_error.emit(video_path, str(e))

            # Processing complete
            print(
                f"Found {len(asl_videos)} ASL videos out of {len(video_files)} total videos"
            )
            if already_processed_count > 0:
                print(f"Skipped {already_processed_count} already processed videos")

            self.processing_complete.emit(asl_videos)

        except Exception as e:
            import traceback

            traceback.print_exc()
            self.processing_error.emit("", str(e))

    def process_landmarks(self, landmark_file, model_data):
        """Process landmarks using a windowing approach"""
        try:
            # Load the landmark data
            with open(landmark_file, "r") as f:
                data = json.load(f)

            # Get video properties
            fps = data.get("video_info", {}).get("fps", 30.0)
            frames = data.get("frames", [])

            if not frames:
                print(f"No frames found in landmark data for {landmark_file}")
                return False, 0.0

            # Process using windows
            window_size_sec = 3.0  # 3 second windows
            overlap_ratio = 0.5  # 50% overlap

            window_frames = int(window_size_sec * fps)
            step_size = int(window_frames * (1 - overlap_ratio))
            step_size = max(1, step_size)

            window_predictions = []

            # Process each window
            for i in range(0, len(frames), step_size):
                end_idx = min(i + window_frames, len(frames))
                if end_idx - i < window_frames // 2:
                    continue

                window_data = frames[i:end_idx]

                # Extract features for this window
                features = extract_features(window_data, fps)

                # Skip windows with insufficient hand data
                if features.get("hand_presence_ratio", 0) < 0.1:
                    continue

                # Prepare features for prediction
                feature_dict = {
                    name: features.get(name, 0) for name in model_data["feature_names"]
                }

                # Convert to DataFrame for consistent format
                feature_df = pd.DataFrame([feature_dict])

                # Scale features
                feature_vector_scaled = model_data["scaler"].transform(feature_df)

                # Make prediction for this window
                prob = model_data["model"].predict_proba(feature_vector_scaled)[0, 1]
                window_predictions.append(prob)

            if not window_predictions:
                print(f"No valid windows for ASL detection in {landmark_file}")
                return False, 0.0

            # Calculate overall prediction
            avg_prediction = sum(window_predictions) / len(window_predictions)
            max_prediction = max(window_predictions)

            # Debug output
            print(
                f"Video analysis: windows={len(window_predictions)}, avg={avg_prediction:.2f}, max={max_prediction:.2f}"
            )

            # Determine if video contains ASL (use Python native bool)
            contains_asl = bool(avg_prediction >= 0.5)

            return contains_asl, float(avg_prediction)

        except Exception as e:
            import traceback

            print(f"Error processing landmarks: {e}")
            traceback.print_exc()
            return False, 0.0


class VideoAnnotationArea(QWidget):
    """Custom widget for video display and annotation"""

    annotation_created = pyqtSignal(dict)  # Emits when an annotation is created
    scrub_position_changed = pyqtSignal(float)  # Emits when scrubbing position changes

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMouseTracking(True)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMinimumSize(640, 480)

        # Video and timeline state
        self.frame = None
        self.video_width = 0
        self.video_height = 0
        self.current_frame = 0
        self.total_frames = 0
        self.fps = 30

        # Annotation variables
        self.annotations = []
        self.is_drawing = False
        self.annotation_start_time = None
        self.annotation_end_time = None
        self.annotation_start_x = 0
        self.current_mouse_pos = (0, 0)

        # Set background color to dark gray
        self.setStyleSheet("background-color: #171717;")

    def set_video_info(self, total_frames, fps):
        """Update video information"""
        self.total_frames = total_frames
        self.fps = fps

    def set_annotations(self, annotations):
        """Set the current annotations"""
        self.annotations = annotations
        self.update()

    def set_frame(self, frame, frame_number):
        """Set the current video frame"""
        self.frame = frame
        self.current_frame = frame_number
        if frame is not None:
            self.video_height, self.video_width = frame.shape[:2]
        self.update()

    def frame_to_time(self, frame_num):
        """Convert frame number to timestamp"""
        return frame_num / self.fps if self.fps > 0 else 0

    def time_to_frame(self, time_sec):
        """Convert timestamp to frame number"""
        return int(time_sec * self.fps)

    def time_to_x(self, time_sec):
        """Convert timestamp to x position in the timeline area"""
        if self.total_frames <= 0 or self.fps <= 0:
            return 0

        timeline_width = self.width()
        total_duration = self.total_frames / self.fps
        return int((time_sec / total_duration) * timeline_width)

    def x_to_time(self, x_pos):
        """Convert x position to timestamp"""
        if self.total_frames <= 0 or self.fps <= 0:
            return 0

        timeline_width = self.width()
        total_duration = self.total_frames / self.fps
        return (x_pos / timeline_width) * total_duration

    def timeline_height(self):
        """Return the height of the timeline area"""
        return 60

    def timeline_rect(self):
        """Return the rectangle of the timeline area"""
        return QRect(
            0,
            self.height() - self.timeline_height(),
            self.width(),
            self.timeline_height(),
        )

    def mousePressEvent(self, event):
        """Handle mouse press events for annotation creation"""
        timeline_rect = self.timeline_rect()

        if event.button() == Qt.LeftButton and timeline_rect.contains(event.pos()):
            # Start annotation creation
            self.is_drawing = True
            self.annotation_start_x = event.x()
            self.annotation_start_time = self.x_to_time(event.x())
            self.annotation_end_time = self.annotation_start_time
            self.current_mouse_pos = (event.x(), event.y())
            self.update()

    def mouseMoveEvent(self, event):
        """Handle mouse move events for annotation creation"""
        self.current_mouse_pos = (event.x(), event.y())

        if self.is_drawing:
            # Update end time of annotation being drawn
            self.annotation_end_time = self.x_to_time(event.x())

            # Emit signal to update video position during dragging
            current_time = self.annotation_end_time
            self.scrub_position_changed.emit(current_time)

            self.update()

    def mouseReleaseEvent(self, event):
        """Handle mouse release events for annotation creation"""
        if event.button() == Qt.LeftButton and self.is_drawing:
            self.is_drawing = False

            # Finalize annotation
            if self.annotation_start_time is None or self.annotation_end_time is None:
                return
            start_time = min(self.annotation_start_time, self.annotation_end_time)
            end_time = max(self.annotation_start_time, self.annotation_end_time)

            # Only create annotation if it has some duration
            if abs(end_time - start_time) > 0.2:  # Minimum 0.2 seconds
                # Get annotation text from custom dialog
                dialog = AnnotationDialog(self)
                if dialog.exec_() == QDialog.Accepted:
                    annotation_text = dialog.get_text()

                    new_annotation = {
                        "start_time": start_time,
                        "end_time": end_time,
                        "start_frame": self.time_to_frame(start_time),
                        "end_frame": self.time_to_frame(end_time),
                        "text": annotation_text,
                    }

                    # Instead of adding here, just emit signal
                    # and let the parent widget add it once
                    self.annotation_created.emit(new_annotation)

            # Reset annotation drawing state
            self.annotation_start_time = None
            self.annotation_end_time = None

    def paintEvent(self, event):
        """Draw the video frame and timeline with annotations"""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        # Calculate video display size
        if self.frame is not None:
            # Calculate video display area
            video_area_height = self.height() - self.timeline_height()

            # Maintain aspect ratio
            aspect_ratio = (
                self.video_width / self.video_height if self.video_height > 0 else 1.33
            )

            if self.width() / video_area_height > aspect_ratio:
                # Width is the limiting factor
                display_height = video_area_height
                display_width = int(display_height * aspect_ratio)
            else:
                # Height is the limiting factor
                display_width = self.width()
                display_height = int(display_width / aspect_ratio)

            # Center the video
            x_offset = (self.width() - display_width) // 2
            y_offset = (video_area_height - display_height) // 2

            # Convert OpenCV BGR to RGB
            rgb_frame = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)
            h, w = rgb_frame.shape[:2]

            # Create QImage and QPixmap
            bytes_per_line = 3 * w
            q_img = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(q_img)

            # Draw the video frame
            target_rect = QRect(x_offset, y_offset, display_width, display_height)
            painter.drawPixmap(target_rect, pixmap)

            # Draw timeline base (background)
            timeline_rect = self.timeline_rect()
            painter.fillRect(timeline_rect, QColor(30, 30, 30))

            # Draw timeline progress
            if self.total_frames > 0:
                progress_width = int(
                    (self.current_frame / self.total_frames) * self.width()
                )
                progress_rect = QRect(
                    0, timeline_rect.y(), progress_width, timeline_rect.height()
                )
                painter.fillRect(progress_rect, QColor(60, 60, 60))

            # Draw timeline cursor (playhead)
            if self.total_frames > 0:
                cursor_x = int((self.current_frame / self.total_frames) * self.width())
                painter.setPen(QPen(QColor(255, 100, 0), 2))
                painter.drawLine(
                    cursor_x,
                    timeline_rect.y(),
                    cursor_x,
                    timeline_rect.y() + timeline_rect.height(),
                )

            # Draw existing annotations
            for annotation in self.annotations:
                start_x = self.time_to_x(annotation["start_time"])
                end_x = self.time_to_x(annotation["end_time"])

                # Draw annotation region
                annotation_rect = QRect(
                    start_x, timeline_rect.y(), end_x - start_x, timeline_rect.height()
                )
                painter.fillRect(annotation_rect, QColor(0, 120, 215, 100))

                # Draw annotation borders
                painter.setPen(QPen(QColor(0, 120, 215), 2))
                painter.drawRect(annotation_rect)

                # Draw label with text in the annotation (if space permits)
                if end_x - start_x > 50:  # Only if there's enough space
                    text = annotation.get("text", "")
                    if text:
                        painter.setPen(QColor(255, 255, 255))
                        text_rect = QRect(
                            start_x + 4,
                            timeline_rect.y() + 4,
                            end_x - start_x - 8,
                            timeline_rect.height() - 8,
                        )
                        painter.drawText(
                            text_rect, Qt.AlignLeft | Qt.AlignTop, text[:20]
                        )

            # Draw annotation being created (if any)
            if self.is_drawing and self.annotation_start_time is not None:
                start_x = self.time_to_x(self.annotation_start_time)
                end_x = self.current_mouse_pos[0]

                # Draw region
                drawing_rect = QRect(
                    min(start_x, end_x),
                    timeline_rect.y(),
                    abs(end_x - start_x),
                    timeline_rect.height(),
                )
                painter.fillRect(drawing_rect, QColor(255, 140, 0, 120))

                # Draw border
                painter.setPen(QPen(QColor(255, 140, 0), 2, Qt.DashLine))
                painter.drawRect(drawing_rect)
        else:
            # If no frame is loaded, just draw placeholder
            painter.setPen(QColor(120, 120, 120))
            painter.drawText(self.rect(), Qt.AlignCenter, "No video loaded")


class VideoLibraryItem(QWidget):
    """Custom widget for video items in the library view"""

    def __init__(self, video_path, parent=None):
        super().__init__(parent)
        self.video_path = video_path
        self.thumbnail = None
        self.video_name = os.path.basename(video_path)

        # Extract thumbnail
        self.extract_thumbnail()

        # Set fixed height but expandable width
        self.setMinimumHeight(120)
        self.setMaximumHeight(120)

    def extract_thumbnail(self):
        """Extract a thumbnail from the video"""
        try:
            cap = cv2.VideoCapture(self.video_path)
            if cap.isOpened():
                # Jump to 1/3 of the video for thumbnail
                frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count // 3)
                ret, frame = cap.read()
                if ret:
                    # Resize thumbnail
                    h, w = frame.shape[:2]
                    aspect = w / h
                    thumb_h = 100
                    thumb_w = int(thumb_h * aspect)
                    self.thumbnail = cv2.resize(frame, (thumb_w, thumb_h))
                cap.release()
        except Exception as e:
            print(f"Error extracting thumbnail: {e}")

    def paintEvent(self, event):
        """Draw the video item with thumbnail and info"""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        # Draw background
        painter.fillRect(self.rect(), QColor(40, 40, 40))

        # Draw border
        painter.setPen(QPen(QColor(60, 60, 60), 1))
        painter.drawRect(QRect(0, 0, self.width() - 1, self.height() - 1))

        # Draw thumbnail
        if self.thumbnail is not None:
            h, w = self.thumbnail.shape[:2]
            thumb_x = 10
            thumb_y = 10

            # Convert OpenCV BGR to RGB
            rgb_frame = cv2.cvtColor(self.thumbnail, cv2.COLOR_BGR2RGB)
            bytes_per_line = 3 * w
            q_img = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(q_img)

            painter.drawPixmap(thumb_x, thumb_y, pixmap)

            # Draw video name
            text_x = thumb_x + w + 15
            text_y = self.height() // 2
            painter.setPen(QColor(220, 220, 220))
            painter.drawText(text_x, text_y, self.video_name)


class ASLAnnotator(QMainWindow):
    """Main application window for ASL video processing and annotation"""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("ASL Video Processor and Annotator")
        self.setMinimumSize(1000, 700)

        # Set application style to dark theme
        self.setStyleSheet("""
            QMainWindow, QWidget {
                background-color: #1e1e1e;
                color: #e0e0e0;
            }
            QPushButton {
                background-color: #2d2d2d;
                color: #e0e0e0;
                border: 1px solid #3d3d3d;
                padding: 6px 12px;
                border-radius: 2px;
            }
            QPushButton:hover {
                background-color: #3d3d3d;
            }
            QPushButton:pressed {
                background-color: #505050;
            }
            QProgressBar {
                border: 1px solid #3d3d3d;
                border-radius: 2px;
                text-align: center;
            }
            QProgressBar::chunk {
                background-color: #0078d7;
            }
            QLineEdit, QListWidget, QTextEdit {
                background-color: #2d2d2d;
                border: 1px solid #3d3d3d;
                color: #e0e0e0;
            }
            QSlider::groove:horizontal {
                background: #3d3d3d;
                height: 8px;
                border-radius: 2px;
            }
            QSlider::handle:horizontal {
                background: #0078d7;
                width: 16px;
                margin: -4px 0;
                border-radius: 8px;
            }
        """)

        # Initialize variables
        self.current_folder = None
        self.output_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "output"
        )
        self.model_path = "asl_model_xgboost.pkl"
        self.mediapipe_model_path = "./tasks/hand_landmarker.task"

        # Video playback variables
        self.video_capture = None
        self.is_playing = False
        self.current_frame = 0
        self.total_frames = 0
        self.fps = 30
        self.current_video_path = None
        self.is_seeking = False  # Flag to indicate seeking operation

        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)

        # Main stacked layout to switch between screens
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QVBoxLayout(self.central_widget)

        # Create all screens
        self.create_folder_selection_screen()
        self.create_processing_screen()
        self.create_library_screen()
        self.create_annotation_screen()

        # Initialize with folder selection screen
        self.current_screen = None
        self.switch_to_folder_screen()

        # Setup playback timer
        self.playback_timer = QTimer(self)
        self.playback_timer.timeout.connect(self.update_video_frame)

        # Load model
        try:
            self.model_data = load_model(self.model_path)
        except Exception as e:
            QMessageBox.warning(
                self, "Model Loading Error", f"Could not load model: {str(e)}"
            )

    def create_folder_selection_screen(self):
        """Create the initial folder selection screen"""
        self.folder_screen = QWidget()
        layout = QVBoxLayout(self.folder_screen)
        layout.setContentsMargins(20, 20, 20, 20)

        # Add some spacing at the top
        layout.addSpacing(80)

        # Title
        title_label = QLabel("ASL Video Processor")
        title_label.setStyleSheet("font-size: 24px; font-weight: bold;")
        title_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(title_label)

        # Subtitle
        subtitle_label = QLabel("Select a folder containing videos to process")
        subtitle_label.setStyleSheet("font-size: 16px;")
        subtitle_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(subtitle_label)

        layout.addSpacing(40)

        # Select folder button
        select_button = QPushButton("Select Folder")
        select_button.setMinimumHeight(40)
        select_button.setMaximumWidth(200)
        select_button.clicked.connect(self.select_folder)

        # Center the button
        button_layout = QHBoxLayout()
        button_layout.addStretch()
        button_layout.addWidget(select_button)
        button_layout.addStretch()
        layout.addLayout(button_layout)

        layout.addStretch()

    def create_processing_screen(self):
        """Create the processing screen with progress indicators"""
        self.processing_screen = QWidget()
        layout = QVBoxLayout(self.processing_screen)
        layout.setContentsMargins(20, 20, 20, 20)

        # Add spacing
        layout.addSpacing(40)

        # Title
        processing_label = QLabel("Processing Videos")
        processing_label.setStyleSheet("font-size: 20px; font-weight: bold;")
        processing_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(processing_label)

        layout.addSpacing(20)

        # Current operation label
        self.current_operation_label = QLabel("Analyzing videos for ASL content...")
        self.current_operation_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.current_operation_label)

        layout.addSpacing(30)

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setMinimumHeight(20)
        self.progress_bar.setTextVisible(True)
        layout.addWidget(self.progress_bar)

        layout.addSpacing(20)

        # Status area
        self.processing_status = QLabel("Initializing...")
        self.processing_status.setStyleSheet("color: #a0a0a0;")
        self.processing_status.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.processing_status)

        # Results summary (will be shown after processing)
        self.results_frame = QFrame()
        self.results_frame.setVisible(False)
        results_layout = QVBoxLayout(self.results_frame)

        self.results_label = QLabel()
        self.results_label.setAlignment(Qt.AlignCenter)
        self.results_label.setStyleSheet("font-size: 16px;")
        results_layout.addWidget(self.results_label)

        results_layout.addSpacing(20)

        # Continue button
        self.continue_button = QPushButton("Continue to Library")
        self.continue_button.setMinimumHeight(40)
        self.continue_button.setMaximumWidth(200)
        self.continue_button.clicked.connect(self.switch_to_library_screen)

        # Center the button
        button_layout = QHBoxLayout()
        button_layout.addStretch()
        button_layout.addWidget(self.continue_button)
        button_layout.addStretch()
        results_layout.addLayout(button_layout)

        layout.addWidget(self.results_frame)
        layout.addStretch()

    def upload_to_aws(self):
        """Dummy function to simulate uploading to AWS"""
        # Show a message indicating upload started
        self.library_status.setText("Uploading to AWS...")

        # Simulate upload delay with a timer
        def upload_complete():
            self.library_status.setText("Upload to AWS completed successfully!")
            QMessageBox.information(
                self, "Upload Complete", "Videos successfully uploaded to AWS."
            )

        # Simulate a delay of 2 seconds
        QTimer.singleShot(2000, upload_complete)

    def create_library_screen(self):
        """Create the video library screen"""
        self.library_screen = QWidget()
        layout = QVBoxLayout(self.library_screen)
        layout.setContentsMargins(10, 10, 10, 10)

        # Header with title and back button
        header_layout = QHBoxLayout()

        self.back_to_folder_button = QPushButton("Back")
        self.back_to_folder_button.clicked.connect(self.switch_to_folder_screen)
        header_layout.addWidget(self.back_to_folder_button)

        library_title = QLabel("ASL Video Library")
        library_title.setStyleSheet("font-size: 18px; font-weight: bold;")
        library_title.setAlignment(Qt.AlignCenter)
        header_layout.addWidget(library_title, 1)  # Stretch factor to center

        # Add upload button
        upload_button = QPushButton("Upload to AWS")
        upload_button.clicked.connect(self.upload_to_aws)
        header_layout.addWidget(upload_button)

        process_new_button = QPushButton("Process New Folder")
        process_new_button.clicked.connect(self.switch_to_folder_screen)
        header_layout.addWidget(process_new_button)

        layout.addLayout(header_layout)

        # Add processing progress section (initially hidden)
        self.library_processing_frame = QFrame()
        self.library_processing_frame.setVisible(False)
        processing_layout = QVBoxLayout(self.library_processing_frame)

        # Current operation label
        self.library_operation_label = QLabel("Analyzing videos for ASL content...")
        self.library_operation_label.setAlignment(Qt.AlignCenter)
        processing_layout.addWidget(self.library_operation_label)

        # Progress bar
        self.library_progress_bar = QProgressBar()
        self.library_progress_bar.setMinimumHeight(20)
        self.library_progress_bar.setTextVisible(True)
        processing_layout.addWidget(self.library_progress_bar)

        # Status area
        self.library_status = QLabel("Initializing...")
        self.library_status.setStyleSheet("color: #a0a0a0;")
        self.library_status.setAlignment(Qt.AlignCenter)
        processing_layout.addWidget(self.library_status)

        layout.addWidget(self.library_processing_frame)

        # Create scrollable area for video items
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setFrameShape(QFrame.NoFrame)

        self.library_content = QWidget()
        self.library_layout = QVBoxLayout(self.library_content)
        self.library_layout.setAlignment(Qt.AlignTop)
        self.library_layout.setSpacing(10)

        scroll_area.setWidget(self.library_content)
        layout.addWidget(scroll_area)

    def create_annotation_screen(self):
        """Create the video annotation screen"""
        self.annotation_screen = QWidget()
        layout = QVBoxLayout(self.annotation_screen)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(5)

        # Header with title and back button
        header_layout = QHBoxLayout()

        back_button = QPushButton("Back to Library")
        back_button.clicked.connect(self.return_to_library)
        header_layout.addWidget(back_button)

        self.video_title_label = QLabel("Video Title")
        self.video_title_label.setStyleSheet("font-size: 16px; font-weight: bold;")
        self.video_title_label.setAlignment(Qt.AlignCenter)
        header_layout.addWidget(self.video_title_label, 1)

        self.save_status_label = QLabel("")
        self.save_status_label.setStyleSheet("color: #a0a0a0;")
        header_layout.addWidget(self.save_status_label)

        layout.addLayout(header_layout)

        # Main split between video/timeline and annotation list
        splitter = QSplitter(Qt.Horizontal)

        # Video and timeline area
        video_container = QWidget()
        video_layout = QVBoxLayout(video_container)
        video_layout.setContentsMargins(0, 0, 0, 0)

        # Video display with annotation timeline
        self.video_annotation_area = VideoAnnotationArea()
        self.video_annotation_area.annotation_created.connect(self.annotation_created)
        self.video_annotation_area.scrub_position_changed.connect(
            self.scrub_to_position
        )
        video_layout.addWidget(self.video_annotation_area, 1)

        # Playback controls
        controls_layout = QHBoxLayout()

        self.play_button = QPushButton("Play")
        self.play_button.clicked.connect(self.toggle_playback)
        controls_layout.addWidget(self.play_button)

        self.position_slider = QSlider(Qt.Horizontal)
        self.position_slider.setMinimum(0)
        self.position_slider.setMaximum(1000)
        self.position_slider.sliderMoved.connect(self.slider_moved)
        self.position_slider.sliderPressed.connect(self.slider_pressed)
        self.position_slider.sliderReleased.connect(self.slider_released)
        controls_layout.addWidget(self.position_slider, 1)

        self.time_label = QLabel("0:00 / 0:00")
        controls_layout.addWidget(self.time_label)

        video_layout.addLayout(controls_layout)

        # Annotation list panel
        annotation_panel = QWidget()
        annotation_layout = QVBoxLayout(annotation_panel)

        panel_title = QLabel("Annotations")
        panel_title.setStyleSheet("font-size: 14px; font-weight: bold;")
        annotation_layout.addWidget(panel_title)

        # Annotation list
        self.annotation_list = QListWidget()
        self.annotation_list.setAlternatingRowColors(True)
        self.annotation_list.itemDoubleClicked.connect(self.edit_annotation)
        annotation_layout.addWidget(self.annotation_list)

        # Annotation controls
        annotation_controls = QHBoxLayout()

        add_annotation_button = QPushButton("Add")
        add_annotation_button.setToolTip("Add annotation (or drag in timeline)")
        add_annotation_button.clicked.connect(self.add_annotation)
        annotation_controls.addWidget(add_annotation_button)

        edit_annotation_button = QPushButton("Edit")
        edit_annotation_button.clicked.connect(self.edit_selected_annotation)
        annotation_controls.addWidget(edit_annotation_button)

        delete_annotation_button = QPushButton("Delete")
        delete_annotation_button.clicked.connect(self.delete_selected_annotation)
        annotation_controls.addWidget(delete_annotation_button)

        annotation_layout.addLayout(annotation_controls)

        # Add widgets to splitter
        splitter.addWidget(video_container)
        splitter.addWidget(annotation_panel)

        # Set initial sizes (70% video, 30% annotation list)
        splitter.setSizes([700, 300])

        layout.addWidget(splitter, 1)

        # Add shortcuts
        self.shortcut_space = QShortcut(QKeySequence(Qt.Key_Space), self)
        self.shortcut_space.activated.connect(self.toggle_playback)

        self.shortcut_left = QShortcut(QKeySequence(Qt.Key_Left), self)
        self.shortcut_left.activated.connect(self.previous_frame)

        self.shortcut_right = QShortcut(QKeySequence(Qt.Key_Right), self)
        self.shortcut_right.activated.connect(self.next_frame)

    def switch_to_folder_screen(self):
        """Switch to the folder selection screen"""
        if self.current_screen is not None:
            self.main_layout.removeWidget(self.current_screen)
            self.current_screen.hide()

        self.main_layout.addWidget(self.folder_screen)
        self.folder_screen.show()
        self.current_screen = self.folder_screen

    def switch_to_processing_screen(self):
        """Switch to the processing screen"""
        if self.current_screen is not None:
            self.main_layout.removeWidget(self.current_screen)
            self.current_screen.hide()

        self.main_layout.addWidget(self.processing_screen)
        self.processing_screen.show()
        self.current_screen = self.processing_screen

        # Reset processing UI elements
        self.progress_bar.setValue(0)
        self.processing_status.setText("Initializing...")
        self.results_frame.setVisible(False)

    def switch_to_library_screen(self):
        """Switch to the video library screen"""
        if self.current_screen is not None:
            self.main_layout.removeWidget(self.current_screen)
            self.current_screen.hide()

        self.main_layout.addWidget(self.library_screen)
        self.library_screen.show()
        self.current_screen = self.library_screen

    def switch_to_annotation_screen(self):
        """Switch to the video annotation screen"""
        if self.current_screen is not None:
            self.main_layout.removeWidget(self.current_screen)
            self.current_screen.hide()

        self.main_layout.addWidget(self.annotation_screen)
        self.annotation_screen.show()
        self.current_screen = self.annotation_screen

    def select_folder(self):
        """Open folder dialog and start processing"""
        folder_dialog = QFileDialog()
        folder_path = folder_dialog.getExistingDirectory(
            self, "Select Folder Containing Videos"
        )

        if folder_path:
            self.current_folder = folder_path
            self.process_folder(folder_path)

    def process_folder(self, folder_path):
        """Start processing all videos in the folder"""
        self.current_folder = folder_path

        # Switch to library screen with progress bar visible
        self.switch_to_library_screen()
        self.library_processing_frame.setVisible(True)
        self.library_progress_bar.setValue(0)
        self.library_operation_label.setText("Analyzing videos for ASL content...")
        self.library_status.setText("Initializing...")

        # Create processing thread
        self.processing_thread = BatchProcessingThread(
            folder_path, self.model_path, self.mediapipe_model_path, self.output_dir
        )

        # Connect signals to library screen elements
        self.processing_thread.progress_update.connect(self.update_library_progress)
        self.processing_thread.video_processed.connect(self.video_processed_library)
        self.processing_thread.processing_complete.connect(
            self.processing_finished_library
        )
        self.processing_thread.processing_error.connect(self.processing_error_library)

        self.processed_videos = []
        self.asl_videos = []

        # Start processing thread
        self.processing_thread.start()

    def update_progress(self, current, total):
        """Update the progress bar during processing"""
        progress_percent = int(100 * current / total) if total > 0 else 0
        self.progress_bar.setValue(progress_percent)
        self.processing_status.setText(f"Processing video {current + 1} of {total}")

    def video_processed(self, video_path, contains_asl):
        """Handle completion of a single video"""
        video_name = os.path.basename(video_path)
        self.processed_videos.append(video_path)

        if contains_asl:
            self.asl_videos.append(video_path)
            self.processing_status.setText(f"ASL detected in {video_name}")
        else:
            self.processing_status.setText(f"No ASL detected in {video_name}")

    def update_library_progress(self, current, total):
        """Update the progress bar in library screen during processing"""
        progress_percent = int(100 * current / total) if total > 0 else 0
        self.library_progress_bar.setValue(progress_percent)
        self.library_status.setText(f"Processing video {current + 1} of {total}")

    def video_processed_library(self, video_path, contains_asl):
        """Handle completion of a single video in library view"""
        video_name = os.path.basename(video_path)
        self.processed_videos.append(video_path)

        if contains_asl:
            self.asl_videos.append(video_path)
            self.library_status.setText(f"ASL detected in {video_name}")

            # Add the video to the library immediately
            self.add_video_to_library(video_path)
        else:
            self.library_status.setText(f"No ASL detected in {video_name}")

    def add_video_to_library(self, video_path):
        """Add a single video to the library"""
        video_item = VideoLibraryItem(video_path)
        video_item.setMinimumWidth(self.library_content.width() - 30)
        video_item.setCursor(Qt.PointingHandCursor)
        video_item.mousePressEvent = lambda event, path=video_path: self.open_video(
            path
        )
        self.library_layout.insertWidget(0, video_item)  # Add to top of list

    def processing_error_library(self, video_path, error_message):
        """Handle processing errors in library view"""
        if video_path:
            video_name = os.path.basename(video_path)
            self.library_status.setText(
                f"Error processing {video_name}: {error_message}"
            )
        else:
            self.library_status.setText(f"Processing error: {error_message}")

    def processing_finished_library(self, asl_videos):
        """Handle completion of all processing in library view"""
        self.library_progress_bar.setValue(100)
        self.asl_videos = asl_videos

        # Display completion status
        total_videos = len(self.processed_videos)
        asl_count = len(self.asl_videos)
        self.library_status.setText(
            f"Processed {total_videos} videos. Found {asl_count} videos containing ASL."
        )

        # Hide the progress section after a delay
        QTimer.singleShot(3000, lambda: self.library_processing_frame.setVisible(False))

    def processing_error(self, video_path, error_message):
        """Handle processing errors"""
        if video_path:
            video_name = os.path.basename(video_path)
            self.processing_status.setText(
                f"Error processing {video_name}: {error_message}"
            )
        else:
            self.processing_status.setText(f"Processing error: {error_message}")

    def processing_finished(self, asl_videos):
        """Handle completion of all processing"""
        self.progress_bar.setValue(100)
        self.asl_videos = asl_videos

        # Display results
        total_videos = len(self.processed_videos)
        asl_count = len(self.asl_videos)

        self.results_label.setText(
            f"Processed {total_videos} videos\nFound {asl_count} videos containing ASL"
        )

        self.results_frame.setVisible(True)
        self.processing_status.setText("Processing complete!")

        # Populate the library screen
        self.populate_library(self.asl_videos)

    def populate_library(self, video_list):
        """Fill the library with video items"""
        # Clear existing items
        for i in reversed(range(self.library_layout.count())):
            widget = self.library_layout.itemAt(i).widget()
            if widget:
                widget.deleteLater()

        if not video_list:
            # Add a "no videos" message
            no_videos_label = QLabel(
                "No ASL videos found. Try processing another folder."
            )
            no_videos_label.setAlignment(Qt.AlignCenter)
            no_videos_label.setStyleSheet("color: #a0a0a0; padding: 40px;")
            self.library_layout.addWidget(no_videos_label)
            return

        # Add video items
        for video_path in video_list:
            video_item = VideoLibraryItem(video_path)
            video_item.setMinimumWidth(self.library_content.width() - 30)
            video_item.setCursor(Qt.PointingHandCursor)
            video_item.mousePressEvent = lambda event, path=video_path: self.open_video(
                path
            )
            self.library_layout.addWidget(video_item)

        # Add some spacing at the end
        self.library_layout.addStretch()

    def open_video(self, video_path):
        """Open a video for annotation"""
        self.current_video_path = video_path

        # Update title
        self.video_title_label.setText(os.path.basename(video_path))

        # Load annotations if they exist
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        annotation_path = os.path.join(
            self.output_dir, f"{video_name}_annotations.json"
        )

        self.annotations = []
        if os.path.exists(annotation_path):
            try:
                with open(annotation_path, "r") as f:
                    annotation_data = json.load(f)
                    self.annotations = annotation_data.get("annotations", [])
            except Exception as e:
                print(f"Error loading annotations: {e}")

        # Open the video
        self.load_video(video_path)

        # Switch to annotation screen
        self.switch_to_annotation_screen()

    def load_video(self, video_path):
        """Load a video for annotation"""
        # Close existing video if open
        if self.video_capture:
            self.video_capture.release()
            self.video_capture = None

        # Stop playback
        if self.is_playing:
            self.playback_timer.stop()
            self.is_playing = False
            self.play_button.setText("Play")

        # Open the video
        self.video_capture = cv2.VideoCapture(video_path)
        if not self.video_capture.isOpened():
            QMessageBox.warning(self, "Error", "Could not open video file.")
            return

        # Get video properties
        self.total_frames = int(self.video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.video_capture.get(cv2.CAP_PROP_FPS)
        if self.fps <= 0:
            self.fps = 30  # Default if can't determine

        # Set position slider range
        self.position_slider.setMaximum(self.total_frames - 1)
        self.position_slider.setValue(0)

        # Update video widget with info
        self.video_annotation_area.set_video_info(self.total_frames, self.fps)
        self.video_annotation_area.set_annotations(self.annotations)

        # Reset position
        self.current_frame = 0
        self.video_capture.set(cv2.CAP_PROP_POS_FRAMES, 0)

        # Read first frame
        ret, frame = self.video_capture.read()
        if ret:
            self.video_annotation_area.set_frame(frame, 0)

        # Update annotation list
        self.update_annotation_list()

        # Update time label
        self.update_time_label()

    def update_annotation_list(self):
        """Update the list of annotations"""
        self.annotation_list.clear()

        # Sort annotations by start time
        sorted_annotations = sorted(self.annotations, key=lambda x: x["start_time"])

        for i, annotation in enumerate(sorted_annotations):
            start_time = self.format_time(annotation["start_time"])
            end_time = self.format_time(annotation["end_time"])
            text = annotation.get("text", "")

            item_text = f"{i + 1}. [{start_time} - {end_time}] {text}"
            self.annotation_list.addItem(item_text)

    def format_time(self, seconds):
        """Format time in seconds to MM:SS.SS format"""
        minutes = int(seconds // 60)
        secs = seconds % 60
        return f"{minutes}:{secs:05.2f}"

    def update_time_label(self):
        """Update the time display label"""
        if self.video_capture and self.fps > 0:
            current_time = self.current_frame / self.fps
            total_time = self.total_frames / self.fps
            self.time_label.setText(
                f"{self.format_time(current_time)} / {self.format_time(total_time)}"
            )

    def toggle_playback(self):
        """Toggle video playback"""
        if not self.video_capture:
            return

        if self.is_playing:
            self.playback_timer.stop()
            self.play_button.setText("Play")
            self.is_playing = False
        else:
            # Calculate timer interval based on FPS
            interval = int(1000 / self.fps)
            self.playback_timer.start(interval)
            self.play_button.setText("Pause")
            self.is_playing = True

    def update_video_frame(self):
        """Update video frame during playback"""
        if not self.video_capture:
            return

        if self.current_frame >= self.total_frames - 1:
            # End of video
            self.playback_timer.stop()
            self.play_button.setText("Play")
            self.is_playing = False
            return

        # Read next frame
        ret, frame = self.video_capture.read()
        if not ret:
            self.playback_timer.stop()
            self.play_button.setText("Play")
            self.is_playing = False
            return

        # Update display
        self.current_frame += 1
        self.video_annotation_area.set_frame(frame, self.current_frame)

        # Update slider without triggering events
        self.position_slider.blockSignals(True)
        self.position_slider.setValue(self.current_frame)
        self.position_slider.blockSignals(False)

        # Update time label
        self.update_time_label()

    def scrub_to_position(self, time_pos):
        """Seek to the given time position (used during annotation dragging)"""
        if not self.video_capture:
            return

        # Convert time to frame position
        frame_pos = int(time_pos * self.fps)
        if frame_pos < 0:
            frame_pos = 0
        if frame_pos >= self.total_frames:
            frame_pos = self.total_frames - 1

        # Set the new position
        self.current_frame = frame_pos
        self.video_capture.set(cv2.CAP_PROP_POS_FRAMES, frame_pos)

        # Read the frame
        ret, frame = self.video_capture.read()
        if ret:
            self.video_annotation_area.set_frame(frame, frame_pos)

        # Update slider position
        self.position_slider.blockSignals(True)
        self.position_slider.setValue(frame_pos)
        self.position_slider.blockSignals(False)

        # Update time label
        self.update_time_label()

    def slider_pressed(self):
        """Handle slider press - pause video if playing"""
        self.is_seeking = True
        if self.is_playing:
            self.playback_timer.stop()

    def slider_moved(self, position):
        """Handle slider movement"""
        if not self.video_capture or not self.is_seeking:
            return

        # Scale the position to match actual frame count
        actual_position = int(
            position * (self.total_frames - 1) / self.position_slider.maximum()
        )

        # Update time label while dragging
        time_pos = actual_position / self.fps
        total_time = self.total_frames / self.fps
        self.time_label.setText(
            f"{self.format_time(time_pos)} / {self.format_time(total_time)}"
        )

        # Seek to the frame and update display during dragging
        self.video_capture.set(cv2.CAP_PROP_POS_FRAMES, actual_position)
        ret, frame = self.video_capture.read()
        if ret:
            self.current_frame = actual_position
            self.video_annotation_area.set_frame(frame, actual_position)

    def slider_released(self):
        """Handle slider release - finalize seek position"""
        self.is_seeking = False
        if not self.video_capture:
            return

        # Resume playback if was playing
        if self.is_playing:
            self.playback_timer.start()

    def previous_frame(self):
        """Go to previous frame"""
        if not self.video_capture or self.current_frame <= 0:
            return

        # Pause if playing
        was_playing = self.is_playing
        if was_playing:
            self.playback_timer.stop()
            self.is_playing = False

        # Go back one frame
        self.current_frame -= 1
        self.video_capture.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame)

        # Read frame
        ret, frame = self.video_capture.read()
        if ret:
            self.video_annotation_area.set_frame(frame, self.current_frame)

        # Update slider
        self.position_slider.setValue(self.current_frame)

        # Update time label
        self.update_time_label()

    def next_frame(self):
        """Go to next frame"""
        if not self.video_capture or self.current_frame >= self.total_frames - 1:
            return

        # Pause if playing
        was_playing = self.is_playing
        if was_playing:
            self.playback_timer.stop()
            self.is_playing = False

        # Go to next frame
        self.current_frame += 1
        self.video_capture.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame)

        # Read frame
        ret, frame = self.video_capture.read()
        if ret:
            self.video_annotation_area.set_frame(frame, self.current_frame)

        # Update slider
        self.position_slider.setValue(self.current_frame)

        # Update time label
        self.update_time_label()

    def add_annotation(self):
        """Add a new annotation at the current position"""
        if not self.video_capture:
            return

        # Pause playback
        was_playing = self.is_playing
        if was_playing:
            self.playback_timer.stop()
            self.is_playing = False

        # Get current time
        current_time = self.current_frame / self.fps

        # Ask for annotation text and duration using custom dialog
        dialog = AnnotationDialog(self)
        if dialog.exec_() == QDialog.Accepted:
            annotation_text = dialog.get_text()

            # Ask for duration
            duration, ok = QInputDialog.getDouble(
                self, "Annotation Duration", "Duration (seconds):", 3.0, 0.5, 30.0, 1
            )

            if ok:
                # Create new annotation
                new_annotation = {
                    "start_time": current_time,
                    "end_time": current_time + duration,
                    "start_frame": self.current_frame,
                    "end_frame": min(
                        self.current_frame + int(duration * self.fps),
                        self.total_frames - 1,
                    ),
                    "text": annotation_text,
                }

                # Add annotation
                self.annotation_created(new_annotation)

    def annotation_created(self, annotation):
        """Handle a new annotation created by dragging or add button"""
        # Add the annotation to our list
        self.annotations.append(annotation)

        # Update the UI
        self.video_annotation_area.set_annotations(self.annotations)
        self.update_annotation_list()

        # Save annotations
        self.save_annotations()

    def edit_annotation(self, item):
        """Edit the annotation when double-clicked in the list"""
        if not item:
            return

        try:
            # Extract index from item text (format: "1. [time] text")
            index = int(item.text().split(".")[0]) - 1
            if 0 <= index < len(self.annotations):
                self.edit_annotation_at_index(index)
        except Exception as e:
            print(f"Error editing annotation: {e}")

    def edit_selected_annotation(self):
        """Edit the currently selected annotation"""
        selected_items = self.annotation_list.selectedItems()
        if not selected_items:
            return

        try:
            # Extract index from item text
            index = int(selected_items[0].text().split(".")[0]) - 1
            if 0 <= index < len(self.annotations):
                self.edit_annotation_at_index(index)
        except Exception as e:
            print(f"Error editing annotation: {e}")

    def edit_annotation_at_index(self, index):
        """Edit the annotation at the specified index"""
        if not 0 <= index < len(self.annotations):
            return

        annotation = self.annotations[index]
        current_text = annotation.get("text", "")

        # Ask for new text using custom dialog
        dialog = AnnotationDialog(self, current_text)
        if dialog.exec_() == QDialog.Accepted:
            new_text = dialog.get_text()
            annotation["text"] = new_text

            # Update UI
            self.update_annotation_list()
            self.video_annotation_area.set_annotations(self.annotations)

            # Save annotations
            self.save_annotations()

    def delete_selected_annotation(self):
        """Delete the selected annotation"""
        selected_items = self.annotation_list.selectedItems()
        if not selected_items:
            return

        try:
            # Extract index from item text
            index = int(selected_items[0].text().split(".")[0]) - 1
            if 0 <= index < len(self.annotations):
                # Ask for confirmation
                confirm = QMessageBox.question(
                    self,
                    "Confirm Deletion",
                    "Are you sure you want to delete this annotation?",
                    QMessageBox.Yes | QMessageBox.No,
                )

                if confirm == QMessageBox.Yes:
                    # Remove annotation
                    del self.annotations[index]

                    # Update UI
                    self.update_annotation_list()
                    self.video_annotation_area.set_annotations(self.annotations)

                    # Save annotations
                    self.save_annotations()
        except Exception as e:
            print(f"Error deleting annotation: {e}")

    def save_annotations(self):
        """Save annotations to a JSON file"""
        if not self.current_video_path:
            return

        video_name = os.path.splitext(os.path.basename(self.current_video_path))[0]
        annotation_path = os.path.join(
            self.output_dir, f"{video_name}_annotations.json"
        )

        # Create data structure
        annotation_data = {
            "video_path": self.current_video_path,
            "annotations": self.annotations,
        }

        # Save to file
        try:
            with open(annotation_path, "w") as f:
                json.dump(annotation_data, f, indent=2)

            # Show temporary save status
            self.save_status_label.setText("Saved ✓")
            QTimer.singleShot(2000, lambda: self.save_status_label.setText(""))

        except Exception as e:
            QMessageBox.warning(
                self, "Save Error", f"Error saving annotations: {str(e)}"
            )

    def return_to_library(self):
        """Return to the library screen"""
        if self.video_capture:
            # Stop playback
            if self.is_playing:
                self.playback_timer.stop()
                self.is_playing = False

            # Release video capture
            self.video_capture.release()
            self.video_capture = None

        # Switch to library screen
        self.switch_to_library_screen()

    def closeEvent(self, event):
        """Handle application close"""
        # Clean up resources
        if self.video_capture:
            self.video_capture.release()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ASLAnnotator()
    window.show()
    sys.exit(app.exec_())
