"""
Face Filter Application using OpenCV (No MediaPipe required)
Works with Python 3.13+
Real-time face detection with filter overlays using Haar Cascades

Author: Face Filter App
Date: October 2025
"""

import cv2
import numpy as np
from datetime import datetime
import os

class FaceFilterApp:
    def __init__(self):
        """Initialize the face filter application"""
        # Load Haar Cascade classifiers for face and eye detection
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        self.eye_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_eye.xml'
        )
        
        # Create filters directory if it doesn't exist
        if not os.path.exists('filters'):
            os.makedirs('filters')
        
        # Create snapshots directory
        if not os.path.exists('snapshots'):
            os.makedirs('snapshots')
        
        # Load filters (will create sample filters if not found)
        self.filters = self.load_filters()
        self.filter_names = list(self.filters.keys())
        self.current_filter_index = 0
        
        # Display settings
        self.show_help = True
        
        print("Face Filter App initialized successfully!")
        print(f"Loaded {len(self.filter_names)} filters")
        
    def create_sample_filters(self):
        """Create sample filter images if they don't exist"""
        print("Creating sample filters...")
        
        # Glasses filter - larger and more visible
        glasses = np.zeros((300, 600, 4), dtype=np.uint8)
        # Left lens
        cv2.circle(glasses, (150, 150), 80, (50, 50, 50, 255), -1)
        cv2.circle(glasses, (150, 150), 70, (200, 200, 255, 200), -1)
        cv2.circle(glasses, (150, 150), 25, (100, 150, 255, 255), -1)
        # Right lens
        cv2.circle(glasses, (450, 150), 80, (50, 50, 50, 255), -1)
        cv2.circle(glasses, (450, 150), 70, (200, 200, 255, 200), -1)
        cv2.circle(glasses, (450, 150), 25, (100, 150, 255, 255), -1)
        # Bridge
        cv2.rectangle(glasses, (230, 140), (370, 160), (50, 50, 50, 255), -1)
        # Temples (arms)
        cv2.rectangle(glasses, (70, 140), (100, 160), (50, 50, 50, 255), -1)
        cv2.rectangle(glasses, (500, 140), (530, 160), (50, 50, 50, 255), -1)
        cv2.imwrite('filters/glasses.png', glasses)
        
        # Crown filter
        crown = np.zeros((300, 500, 4), dtype=np.uint8)
        points = np.array([[50, 250], [125, 80], [175, 250], [250, 50], 
                          [325, 250], [375, 80], [450, 250], [250, 280]])
        cv2.fillPoly(crown, [points], (255, 215, 0, 255))
        cv2.polylines(crown, [points], True, (218, 165, 32, 255), 5)
        # Add jewels
        for x in [125, 250, 375]:
            cv2.circle(crown, (x, 150), 20, (255, 0, 0, 255), -1)
            cv2.circle(crown, (x, 150), 15, (200, 0, 100, 255), -1)
        cv2.imwrite('filters/crown.png', crown)
        
        # Mask filter (medical/surgical style)
        mask = np.zeros((250, 500, 4), dtype=np.uint8)
        cv2.ellipse(mask, (250, 125), (220, 110), 0, 0, 360, (100, 200, 255, 240), -1)
        cv2.ellipse(mask, (250, 125), (220, 110), 0, 0, 360, (50, 150, 200, 255), 4)
        # Add pleats (horizontal lines)
        for y in [80, 100, 120, 140, 160]:
            cv2.line(mask, (50, y), (450, y), (80, 170, 220, 200), 2)
        # Ear loops
        cv2.ellipse(mask, (30, 125), (15, 40), 0, 0, 360, (200, 200, 200, 255), 3)
        cv2.ellipse(mask, (470, 125), (15, 40), 0, 0, 360, (200, 200, 200, 255), 3)
        cv2.imwrite('filters/mask.png', mask)
        
        # Dog ears filter
        dog_ears = np.zeros((400, 600, 4), dtype=np.uint8)
        # Left ear
        left_ear = np.array([[80, 350], [40, 150], [140, 80], [200, 220]])
        cv2.fillPoly(dog_ears, [left_ear], (139, 90, 43, 255))
        cv2.polylines(dog_ears, [left_ear], True, (101, 67, 33, 255), 4)
        # Inner left ear
        inner_left = np.array([[100, 280], [80, 180], [150, 130], [180, 230]])
        cv2.fillPoly(dog_ears, [inner_left], (255, 192, 203, 255))
        # Right ear
        right_ear = np.array([[520, 350], [560, 150], [460, 80], [400, 220]])
        cv2.fillPoly(dog_ears, [right_ear], (139, 90, 43, 255))
        cv2.polylines(dog_ears, [right_ear], True, (101, 67, 33, 255), 4)
        # Inner right ear
        inner_right = np.array([[500, 280], [520, 180], [450, 130], [420, 230]])
        cv2.fillPoly(dog_ears, [inner_right], (255, 192, 203, 255))
        cv2.imwrite('filters/dog_ears.png', dog_ears)
        
        # Mustache filter
        mustache = np.zeros((200, 400, 4), dtype=np.uint8)
        # Left side
        cv2.ellipse(mustache, (120, 100), (100, 60), -20, 0, 180, (20, 20, 20, 255), -1)
        # Right side
        cv2.ellipse(mustache, (280, 100), (100, 60), 20, 0, 180, (20, 20, 20, 255), -1)
        # Add curls at ends
        cv2.circle(mustache, (30, 70), 25, (20, 20, 20, 255), -1)
        cv2.circle(mustache, (370, 70), 25, (20, 20, 20, 255), -1)
        cv2.imwrite('filters/mustache.png', mustache)
        
        print("Sample filters created successfully!")
        
    def load_filters(self):
        """Load filter images from the filters directory"""
        filters = {}
        filter_files = {
            'Glasses': 'filters/glasses.png',
            'Crown': 'filters/crown.png',
            'Mask': 'filters/mask.png',
            'Dog Ears': 'filters/dog_ears.png',
            'Mustache': 'filters/mustache.png'
        }
        
        # Check if filters exist, if not create sample ones
        create_samples = False
        for filename in filter_files.values():
            if not os.path.exists(filename):
                create_samples = True
                break
        
        if create_samples:
            self.create_sample_filters()
        
        # Load all filters
        for name, filename in filter_files.items():
            if os.path.exists(filename):
                img = cv2.imread(filename, cv2.IMREAD_UNCHANGED)
                if img is not None:
                    filters[name] = img
                    print(f"✓ Loaded filter: {name}")
                else:
                    print(f"✗ Failed to load: {filename}")
            else:
                print(f"✗ Filter not found: {filename}")
        
        if not filters:
            print("ERROR: No filters loaded!")
            filters['None'] = None
        
        return filters
    
    def overlay_transparent(self, background, overlay, x, y, w, h):
        """Overlay a transparent PNG image on background"""
        if overlay is None:
            return background
        
        # Ensure coordinates are within bounds
        bg_h, bg_w = background.shape[:2]
        
        # Adjust if overlay goes out of bounds
        if x < 0:
            w += x
            x = 0
        if y < 0:
            h += y
            y = 0
        if x + w > bg_w:
            w = bg_w - x
        if y + h > bg_h:
            h = bg_h - y
        
        if w <= 0 or h <= 0:
            return background
        
        # Resize overlay to fit
        overlay_resize = cv2.resize(overlay, (w, h), interpolation=cv2.INTER_AREA)
        
        # Extract alpha channel if available
        if overlay_resize.shape[2] == 4:
            alpha = overlay_resize[:, :, 3] / 255.0
            alpha = alpha[:, :, np.newaxis]
            
            # Blend the overlay with background
            foreground = overlay_resize[:, :, :3]
            background_section = background[y:y+h, x:x+w]
            
            blended = (alpha * foreground + (1 - alpha) * background_section).astype(np.uint8)
            background[y:y+h, x:x+w] = blended
        else:
            # No alpha channel, just overlay
            background[y:y+h, x:x+w] = overlay_resize[:, :, :3]
        
        return background
    
    def apply_filter(self, frame, face, eyes, filter_name):
        """Apply the selected filter based on detected face and eyes"""
        x, y, w, h = face
        
        if filter_name == 'Glasses' and len(eyes) >= 2:
            # Use detected eyes for glasses positioning
            eye_centers = []
            for (ex, ey, ew, eh) in eyes[:2]:
                eye_centers.append((x + ex + ew//2, y + ey + eh//2))
            
            if len(eye_centers) == 2:
                # Calculate glasses dimensions
                eye_distance = abs(eye_centers[1][0] - eye_centers[0][0])
                glasses_width = int(eye_distance * 2.3)
                glasses_height = int(glasses_width * 0.5)
                
                # Center between eyes
                center_x = (eye_centers[0][0] + eye_centers[1][0]) // 2
                center_y = (eye_centers[0][1] + eye_centers[1][1]) // 2
                
                gx = center_x - glasses_width // 2
                gy = center_y - glasses_height // 2
                
                frame = self.overlay_transparent(frame, self.filters['Glasses'], 
                                                gx, gy, glasses_width, glasses_height)
        
        elif filter_name == 'Crown':
            # Position crown above head
            crown_width = int(w * 1.2)
            crown_height = int(crown_width * 0.6)
            cx = x + w//2 - crown_width//2
            cy = y - int(crown_height * 0.8)
            
            frame = self.overlay_transparent(frame, self.filters['Crown'], 
                                            cx, cy, crown_width, crown_height)
        
        elif filter_name == 'Mask':
            # Position mask over lower face
            mask_width = int(w * 0.9)
            mask_height = int(mask_width * 0.5)
            mx = x + w//2 - mask_width//2
            my = y + int(h * 0.45)
            
            frame = self.overlay_transparent(frame, self.filters['Mask'], 
                                            mx, my, mask_width, mask_height)
        
        elif filter_name == 'Dog Ears':
            # Position ears on top of head
            ears_width = int(w * 1.6)
            ears_height = int(ears_width * 0.65)
            ex = x + w//2 - ears_width//2
            ey = y - int(ears_height * 0.6)
            
            frame = self.overlay_transparent(frame, self.filters['Dog Ears'], 
                                            ex, ey, ears_width, ears_height)
        
        elif filter_name == 'Mustache':
            # Position mustache above mouth (lower part of face)
            mustache_width = int(w * 0.7)
            mustache_height = int(mustache_width * 0.5)
            mx = x + w//2 - mustache_width//2
            my = y + int(h * 0.65)
            
            frame = self.overlay_transparent(frame, self.filters['Mustache'], 
                                            mx, my, mustache_width, mustache_height)
        
        return frame
    
    def draw_help_text(self, frame):
        """Draw help text on frame"""
        help_text = [
            "=== FACE FILTER APP ===",
            "Controls:",
            "1-5: Switch filters",
            "H: Toggle help",
            "S: Save snapshot",
            "Q: Quit",
            "",
            f"Filter: {self.filter_names[self.current_filter_index]}"
        ]
        
        # Semi-transparent background for text
        overlay = frame.copy()
        cv2.rectangle(overlay, (5, 5), (280, 240), (0, 0, 0), -1)
        frame = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)
        
        y_offset = 25
        for i, text in enumerate(help_text):
            cv2.putText(frame, text, (15, y_offset + i * 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(frame, text, (15, y_offset + i * 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
        
        return frame
    
    def save_snapshot(self, frame):
        """Save current frame as snapshot"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"snapshots/snapshot_{timestamp}.png"
        cv2.imwrite(filename, frame)
        print(f"📸 Snapshot saved: {filename}")
        
        # Show notification on frame briefly
        cv2.putText(frame, "SNAPSHOT SAVED!", (frame.shape[1]//2 - 150, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
        return filename
    
    def run(self):
        """Main application loop"""
        print("\n" + "="*50)
        print("🎭 FACE FILTER APPLICATION")
        print("="*50)
        print("Starting webcam...")
        
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("❌ Error: Could not open webcam!")
            print("Please check:")
            print("  - Camera is connected")
            print("  - Camera permissions are enabled")
            print("  - No other app is using the camera")
            return
        
        # Set camera properties for better performance
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        print("✓ Webcam opened successfully!")
        print("\n📋 Controls:")
        print("  1-5: Switch between filters")
        print("  H: Toggle help display")
        print("  S: Save snapshot")
        print("  Q: Quit application")
        print("\n🎬 Press any key to start...\n")
        
        snapshot_saved = False
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("❌ Failed to grab frame")
                break
            
            # Flip frame horizontally for mirror effect
            frame = cv2.flip(frame, 1)
            
            # Convert to grayscale for face detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Detect faces
            faces = self.face_cascade.detectMultiScale(
                gray, 
                scaleFactor=1.1, 
                minNeighbors=5, 
                minSize=(100, 100)
            )
            
            # Process each detected face
            for (x, y, w, h) in faces:
                # Detect eyes within the face region
                roi_gray = gray[y:y+h, x:x+w]
                eyes = self.eye_cascade.detectMultiScale(roi_gray, 1.1, 10)
                
                # Apply current filter
                current_filter = self.filter_names[self.current_filter_index]
                frame = self.apply_filter(frame, (x, y, w, h), eyes, current_filter)
                
                # Optional: Draw face rectangle for debugging (comment out for cleaner look)
                # cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            # Draw help text if enabled
            if self.show_help:
                frame = self.draw_help_text(frame)
            
            # Display frame
            cv2.imshow('Face Filter Application - Press Q to Quit', frame)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                print("👋 Quitting...")
                break
            elif key == ord('1') and len(self.filter_names) > 0:
                self.current_filter_index = 0
                print(f"🎭 Filter: {self.filter_names[0]}")
            elif key == ord('2') and len(self.filter_names) > 1:
                self.current_filter_index = 1
                print(f"🎭 Filter: {self.filter_names[1]}")
            elif key == ord('3') and len(self.filter_names) > 2:
                self.current_filter_index = 2
                print(f"🎭 Filter: {self.filter_names[2]}")
            elif key == ord('4') and len(self.filter_names) > 3:
                self.current_filter_index = 3
                print(f"🎭 Filter: {self.filter_names[3]}")
            elif key == ord('5') and len(self.filter_names) > 4:
                self.current_filter_index = 4
                print(f"🎭 Filter: {self.filter_names[4]}")
            elif key == ord('h'):
                self.show_help = not self.show_help
                print(f"ℹ️  Help display: {'ON' if self.show_help else 'OFF'}")
            elif key == ord('s'):
                self.save_snapshot(frame)
                snapshot_saved = True
        
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        print("\n✅ Application closed successfully!")
        if snapshot_saved:
            print("📁 Check the 'snapshots' folder for your saved images!")


if __name__ == '__main__':
    try:
        app = FaceFilterApp()
        app.run()
    except KeyboardInterrupt:
        print("\n\n⚠️  Application interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("Please make sure opencv-python is installed correctly")