# test_models.py
from ultralytics import YOLO
import cv2

# Load both models
model_general = YOLO("yolov8m.pt")   # Pretrained model (detects knife etc.)
model_custom  = YOLO("best.pt")      # Custom model (detects fight, fire, gun, accident)

# Define emergency labels
emergency_labels = {"knife", "gun", "fire", "accident", "fight"}

# Choose a test video (replace this path with your video)
video_path = "C:\\Users\\bhara\\Downloads\\WhatsApp Video 2025-11-13 at 7.04.18 PM.mp4"   # e.g. a small sample video with fire or fight

cap = cv2.VideoCapture(video_path)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run both models
    results_general = model_general(frame)
    results_custom  = model_custom(frame)

    # Combine both results
    all_results = [*results_general, *results_custom]

    for result in all_results:
        for box in result.boxes:
            cls = int(box.cls[0])
            name = result.names[cls]
            conf = float(box.conf[0])

            # Red box for emergencies, Green for normal
            if name in emergency_labels:
                color = (0, 0, 255)
                label = f"{name} ({conf:.2f}) 🚨"
            else:
                color = (0, 255, 0)
                label = f"{name} ({conf:.2f})"

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    cv2.imshow("Combined Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
