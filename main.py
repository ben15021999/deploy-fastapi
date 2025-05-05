import cv2
import numpy as np
import io
from ultralytics import YOLO
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

model = YOLO('yolo11m-seg.pt')

from fastapi import FastAPI, UploadFile

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.post("/segment")
async def segment(image: UploadFile):
    img_bytes = await image.read()
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")

    results = model.predict(img)

    response = []

    for r in results:
        names = model.names
        for i, box in enumerate(r.boxes):
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            label = names[cls_id]

            # Mask coordinates (numpy array shape: [num_points, 2])
            if r.masks is not None:
                mask = r.masks.data[i].cpu().numpy()  # shape: [H, W]
                # You can convert the mask to polygon or keep it raw
                mask_binary = (mask > 0.5).astype(np.uint8)

                # Optional: extract contours
                contours = cv2.findContours(mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
                polygons = []
                for cnt in contours:
                    polygon = cnt.squeeze().tolist()
                    if isinstance(polygon[0], list):  # multiple points
                        polygons.append(polygon)
                    else:
                        polygons.append([polygon])  # single point
            else:
                polygons = []

            response.append({
                "class_id": cls_id,
                "name": label,
                "confidence": conf,
                "polygons": polygons
            })

    return {"segments": response}

# cap = cv2.VideoCapture(0)

# while cap.isOpened():
#     success, frame = cap.read()
    
#     if success:
#         start = time.perf_counter()

#         results = model(frame)

#         end = time.perf_counter()
        
#         total_time = end - start
#         fps = 1 / total_time


#         annotated_frame = results[0].plot()
#         # cv2.putText(annotated_frame, f'FPS: {int(fps)}')
#         cv2.imshow('Inference', annotated_frame)
#         # Parse results
#         detections = []
#         for r in results:
#             for box in r.boxes:
#                 cls_id = int(box.cls[0])
#                 conf = float(box.conf[0])
#                 xyxy = box.xyxy[0].tolist()
#                 detections.append({
#                     "class_id": cls_id,
#                     "name": model.names[cls_id],
#                     "confidence": conf,
#                     "xmin": xyxy[0],
#                     "ymin": xyxy[1],
#                     "xmax": xyxy[2],
#                     "ymax": xyxy[3],
#                 })
#         print({"detections": detections})
#         cv2.imshow('Inference', annotated_frame)