import cv2
from lerobot.common.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.common.cameras.opencv.camera_opencv import OpenCVCamera
from lerobot.common.cameras.configs import ColorMode, Cv2Rotation

# Construct an `OpenCVCameraConfig` for your laptop webcam
config = OpenCVCameraConfig(
    index_or_path="/dev/video6", # Use the detected ID for clarity, or just 0
    fps=30,                     # Use the detected default FPS
    width=640,                  # Use the detected default width
    height=480,                 # Use the detected default height
    color_mode=ColorMode.RGB,
    rotation=Cv2Rotation.NO_ROTATION
)

# Instantiate and connect an `OpenCVCamera`
camera = OpenCVCamera(config)
try:
    camera.connect() # This will perform a warm-up read by default

    # Read frames asynchronously in a loop
    print("Capturing 100 frames from webcam...")
    for i in range(1000):
        frame = camera.async_read(timeout_ms=200)
        if frame is not None:
            print(f"Async frame {i} shape: {frame.shape}, dtype: {frame.dtype}")
            # Optional: Display the frame using OpenCV (requires `import cv2`)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            cv2.imshow("Webcam Feed", frame_rgb)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            print(f"Async frame {i}: No frame received within timeout. Is the camera busy?")
except Exception as e:
    print(f"An error occurred during camera operation: {e}")
finally:
    camera.disconnect()
    # Optional: Close all OpenCV windows if you used `cv2.imshow`
    if 'cv2' in locals() and cv2.getWindowProperty("Webcam Feed", cv2.WND_PROP_VISIBLE) >= 0:
        cv2.destroyAllWindows()
print("Camera disconnected.")
