from datetime import date
from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks, Request
from fastapi.responses import RedirectResponse, HTMLResponse
from pydantic import BaseModel
from dotenv import load_dotenv
import json
import base64
import os
from PIL import Image
from typing import List, Optional, Any
import io
import asyncio
import numpy as np
import cv2

from ultralytics import YOLO
from ultralytics.nn.tasks import DetectionModel
from ultralytics.nn.modules.conv import Conv, Concat
from ultralytics.nn.modules.block import C3k2, C3, SPPF, Bottleneck, C2f
from ultralytics.nn.modules.head import Detect
import torch.nn
from sqlalchemy import create_engine, Column, Float, String, Integer, ForeignKey
from sqlalchemy.orm import sessionmaker, Session, relationship
from sqlalchemy.ext.declarative import declarative_base
import uuid
import torch.serialization
import traceback

from sqlalchemy.types import JSON

from fastapi.middleware.cors import CORSMiddleware




load_dotenv()
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DATABASE_URL = os.getenv("DATABASE_URL")
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class User(Base):
  __tablename__ = "users"
  id = Column(String, primary_key=True, index=True)
  coins = Column(Integer, default=0)
  activeDays = Column(Integer, default=0)
  lastActive = Column(String, nullable=True) 
  detections = relationship("WasteDetection", back_populates="owner", foreign_keys="[WasteDetection.user_id]")

class WasteDetection(Base):
  __tablename__ = "waste_detections"
  id = Column(String, primary_key=True, index=True)
  base64 = Column(String)
  latitude = Column(Float)
  longitude = Column(Float)
  date_taken = Column(String)
  user_id = Column(String, ForeignKey("users.id"), index=True)
  owner = relationship("User", back_populates="detections", foreign_keys="[WasteDetection.user_id]")
  detected_classes = Column(String)
  status = Column(String, index=True)
  detection_points = Column(JSON, nullable=True)
  collected_by = Column(String, ForeignKey("users.id"), nullable=True, index=True)
  collection_date = Column(String, nullable=True)

Base.metadata.create_all(engine)

def get_db():
  db = SessionLocal()
  try:
    yield db
  finally:
    db.close()
    
class UserActiveRequest(BaseModel):
  userId: str

class JsonResult(BaseModel):
  base64: str
  latitude: float
  longitude: float
  dateTaken: str
  userId: str

class UserResponse(BaseModel):
  id: str
  coins: int
  activeDays: Optional[int] = 0
  lastActive: Optional[str] = None

  class Config:
    orm_mode = True

class WasteDetectionResponse(BaseModel):
  id: str
  base64: str
  latitude: float
  longitude: float
  date_taken: str
  user_id: str
  detected_classes: List[str]
  status: str
  detection_points: Optional[List[Any]] = None

  class Config:
    from_attributes = True

safe_globals_list = [
  DetectionModel, torch.nn.Sequential, Conv, Concat, C3k2, C3, SPPF, Bottleneck, C2f, Detect, torch.nn.Conv2d, torch.nn.BatchNorm2d, torch.nn.ReLU, torch.nn.LeakyReLU, torch.nn.SiLU, torch.nn.Sigmoid, torch.nn.Hardswish, torch.nn.Upsample, torch.nn.MaxPool2d, torch.nn.AdaptiveAvgPool2d, torch.nn.AdaptiveMaxPool2d
  ]

torch.serialization.add_safe_globals(safe_globals_list)

model = YOLO("best.pt")
model_classes = YOLO("best-classes.pt")

def transform_contour_to_original(contour_normalized, crop_x, crop_y, crop_w, crop_h, original_w, original_h):
    """
    Transform contour points from cropped image coordinates to original image coordinates.
    
    Args:
        contour_normalized: List of [x, y] points normalized to 0-1 relative to crop
        crop_x, crop_y: Top-left position of crop in original image (pixels)
        crop_w, crop_h: Width and height of crop (pixels)
        original_w, original_h: Original image dimensions (pixels)
    
    Returns:
        List of [x, y] points normalized to 0-1 relative to original image
    """
    transformed = []
    for point in contour_normalized:
        x_crop_pixels = point[0] * crop_w
        y_crop_pixels = point[1] * crop_h
        
        x_original_pixels = x_crop_pixels + crop_x
        y_original_pixels = y_crop_pixels + crop_y
        
        x_normalized = x_original_pixels / original_w
        y_normalized = y_original_pixels / original_h
        
        transformed.append([x_normalized, y_normalized])
    
    return transformed

def crop_masked_region(image, mask_contour_normalized):
    """
    Crop image using segmentation mask, applying the mask and cropping to bounding box.
    
    Args:
        image: PIL Image (RGB)
        mask_contour_normalized: List of [x, y] normalized points (0-1)
    
    Returns:
        cropped_image: PIL Image of masked and cropped region
        crop_x, crop_y, crop_w, crop_h: Crop position and dimensions in original image
    """
    img_array = np.array(image)
    h, w = img_array.shape[:2]
    
    contour_pixels = np.array([[int(p[0] * w), int(p[1] * h)] for p in mask_contour_normalized], dtype=np.int32)
    
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [contour_pixels], 255)
    
    x, y, box_w, box_h = cv2.boundingRect(contour_pixels)
    
    padding = max(10, int(0.05 * max(box_w, box_h)))
    x = max(0, x - padding)
    y = max(0, y - padding)
    box_w = min(w - x, box_w + 2 * padding)
    box_h = min(h - y, box_h + 2 * padding)
    
    masked_img = cv2.bitwise_and(img_array, img_array, mask=mask)
    
    cropped = masked_img[y:y+box_h, x:x+box_w]
    
    cropped_pil = Image.fromarray(cropped)
    
    min_size = 224
    if cropped_pil.width < min_size or cropped_pil.height < min_size:
        scale = max(min_size / cropped_pil.width, min_size / cropped_pil.height)
        new_w = int(cropped_pil.width * scale)
        new_h = int(cropped_pil.height * scale)
        cropped_pil = cropped_pil.resize((new_w, new_h), Image.LANCZOS)
    
    return cropped_pil, x, y, box_w, box_h

@app.post('/user_active', response_model=UserResponse)
async def record_user_active(request_data: UserActiveRequest, db: Session = Depends(get_db)):
	user_id = request_data.userId
	user = db.query(User).filter(User.id == user_id).first()

	if not user:
		print(f"User {user_id} not found by /user_active, creating new user for activity tracking.")
		user = User(id=user_id, coins=0, activeDays=0, lastActive=None)
		db.add(user)	
	today_str = date.today().isoformat()

	if user.lastActive != today_str:
		user.activeDays = (user.activeDays or 0) + 1
		user.lastActive = today_str
		print(f"User {user.id} marked active for {today_str}. Active days: {user.activeDays}")
	else:
		print(f"User {user.id} already active today ({today_str}). No change to activeDays.")

	try:
		db.commit()
		db.refresh(user)
	except Exception as e:
		db.rollback()
		print(f"!!! DATABASE ERROR ON UPDATING USER ACTIVITY FOR {user.id}: {type(e).__name__} - {str(e)}")
		traceback.print_exc()
		raise HTTPException(status_code=500, detail=f"Failed to update user activity: {str(e)}")

	return UserResponse(
		id=user.id,
		coins=user.coins,
		activeDays=user.activeDays,
		lastActive=user.lastActive
	)


@app.post("/classify")
async def classify_image(data: JsonResult, db: Session = Depends(get_db)):
    if model is None or model_classes is None:
        raise HTTPException(status_code=503, detail="Models not loaded. Check service logs.")
    
    MAX_BASE64_LENGTH = 15_000_000
    if len(data.base64) > MAX_BASE64_LENGTH:
        raise HTTPException(status_code=413, detail="Image is too large. Maximum size is ~10MB.")

    data_for_log = data.model_dump()
    data_for_log["base64"] = f"<image_data_len:{len(data.base64)}>"
    print("Received data for classification:", json.dumps(data_for_log, indent=2))
    
    try:
        img_data = base64.b64decode(data.base64)
        img = Image.open(io.BytesIO(img_data)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid base64 image data: {e}")
    
    try:
        results_stage1 = await asyncio.to_thread(model, img)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Stage 1 inference error: {e}")
    
    lixo_detections = []
    all_sub_classes = []
    class_counts = {"papel": 0, "plastico": 0, "vidro": 0, "metal": 0}
    
    if results_stage1 and results_stage1[0].masks is not None:
        print(f"Stage 1: Found {len(results_stage1[0].masks)} lixo detections")
        
        original_w, original_h = img.size
        
        for lixo_idx, mask in enumerate(results_stage1[0].masks):
            box = results_stage1[0].boxes[lixo_idx]
            class_id = int(box.cls.item())
            class_name = results_stage1[0].names[class_id]
            
            if class_name.lower() == "lixo":
                lixo_contour = mask.xyn[0].tolist()
                
                sub_classes_for_this_lixo = []
                try:
                    cropped_img, crop_x, crop_y, crop_w, crop_h = crop_masked_region(img, lixo_contour)
                    
                    results_stage2 = await asyncio.to_thread(model_classes, cropped_img)
                    
                    if results_stage2 and results_stage2[0].masks is not None:
                        print(f"  Stage 2 for lixo #{lixo_idx}: Found {len(results_stage2[0].masks)} sub-class detections")
                        
                        for sub_idx, sub_mask in enumerate(results_stage2[0].masks):
                            sub_box = results_stage2[0].boxes[sub_idx]
                            sub_class_id = int(sub_box.cls.item())
                            sub_class_name = results_stage2[0].names[sub_class_id].lower()
                            
                            if sub_class_name in ["papel", "plastico", "vidro", "metal"]:
                                sub_contour_crop = sub_mask.xyn[0].tolist()
                                
                                sub_contour_original = transform_contour_to_original(
                                    sub_contour_crop, crop_x, crop_y, crop_w, crop_h,
                                    original_w, original_h
                                )
                                
                                confidence = float(sub_box.conf.item())
                                
                                sub_class_data = {
                                    "class_name": sub_class_name,
                                    "contour": sub_contour_original,
                                    "confidence": confidence
                                }
                                
                                sub_classes_for_this_lixo.append(sub_class_data)
                                all_sub_classes.append(sub_class_data)
                                class_counts[sub_class_name] += 1
                    else:
                        print(f"  Stage 2 for lixo #{lixo_idx}: No sub-classes detected")
                
                except Exception as e:
                    print(f"  Stage 2 failed for lixo #{lixo_idx}: {e}")
                    traceback.print_exc()
                
                lixo_detections.append({
                    "lixo_id": lixo_idx,
                    "lixo_contour": lixo_contour,
                    "sub_classes": sub_classes_for_this_lixo
                })
    else:
        print("Stage 1: No lixo detected")
    
    detection_points_data = {
        "lixo_detections": lixo_detections,
        "class_counts": class_counts
    }
    
    current_status = "Recusada"
    coins_to_award = 0
    
    user = db.query(User).filter(User.id == data.userId).first()
    if not user:
        print(f"User {data.userId} not found, creating new user.")
        user = User(id=data.userId, coins=0)
        db.add(user)
    else:
        print(f"User {user.id} found with {user.coins} coins.")
    
    if len(lixo_detections) > 0:
        current_status = "A coletar"
        coins_to_award = 100
        user.coins += coins_to_award
    elif user.coins >= 50:
        current_status = "Recusada"
        user.coins = max(0, user.coins - 50)
    else:
        current_status = "Recusada"
        user.coins = 0
    
    print(f"User coin balance will be updated to: {user.coins}")
    
    detected_classes_names = ["lixo"] * len(lixo_detections)
    for sub_class in all_sub_classes:
        detected_classes_names.append(sub_class["class_name"])
    
    new_detection_id = str(uuid.uuid4())
    detection_db_entry = WasteDetection(
        id=new_detection_id,
        base64=data.base64,
        latitude=data.latitude,
        longitude=data.longitude,
        date_taken=data.dateTaken,
        user_id=user.id,
        detected_classes=json.dumps(list(set(detected_classes_names))),
        status=current_status,
        detection_points=detection_points_data
    )
    db.add(detection_db_entry)
    
    try:
        print(f"Attempting to commit all changes for user {user.id}...")
        db.commit()
        
        db.refresh(user)
        db.refresh(detection_db_entry)
        
        print("✅ Successfully saved detection with two-stage inference")
        print(
            f"  -> Saved Data: id={detection_db_entry.id}, "
            f"status='{detection_db_entry.status}', "
            f"lixo_count={len(lixo_detections)}"
        )

    except Exception as e:
        db.rollback()
        print(f"!!! DATABASE ERROR: {type(e).__name__} - {str(e)}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")
    
    return {
        "id": new_detection_id,
        "status": "success",
        "message": "Image classified with two-stage inference",
        "detected_classes": detected_classes_names,
        "classification_status": current_status,
        "coins_awarded_this_time": coins_to_award,
        "user_total_coins": user.coins,
        "detection_points": detection_points_data,
        "lixo_count": len(lixo_detections),
        "class_counts": class_counts
    }


@app.get("/users/", response_model=List[UserResponse])
async def get_all_users(skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
	users = db.query(User).offset(skip).limit(limit).all()
	return users

@app.get("/users/{user_id}", response_model=UserResponse)
async def get_user_by_id(user_id: str, db: Session = Depends(get_db)):
	user = db.query(User).filter(User.id == user_id).first()
	if user is None:
		raise HTTPException(status_code=404, detail=f"User not found with id: {user_id}")
	return user

@app.get("/detections/", response_model=List[WasteDetectionResponse])
async def get_all_detections(skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    detections = db.query(WasteDetection).offset(skip).limit(limit).all()
    response_list = []
    for det in detections:
        try:
            parsed_classes = json.loads(det.detected_classes) if det.detected_classes else []
        except json.JSONDecodeError:
            parsed_classes = []
        response_list.append(WasteDetectionResponse(
            id=det.id,
            base64=det.base64,
            latitude=det.latitude,
            longitude=det.longitude,
            date_taken=det.date_taken,
            user_id=det.user_id,
            detected_classes=parsed_classes,
            status=det.status,
            detection_points=det.detection_points
        ))
    return response_list

@app.get("/detections/user/{user_id}", response_model=List[WasteDetectionResponse])
async def get_detections_by_user(user_id: str, skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    detections = db.query(WasteDetection).filter(WasteDetection.user_id == user_id).offset(skip).limit(limit).all()
    if not detections:
        user = db.query(User).filter(User.id == user_id).first()
        if not user:
            raise HTTPException(status_code=404, detail=f"User not found with id: {user_id}")
        return []

    response_list = []
    for det in detections:
        try:
            parsed_classes = json.loads(det.detected_classes) if det.detected_classes else []
        except json.JSONDecodeError:
            parsed_classes = []
        response_list.append(WasteDetectionResponse(
            id=det.id,
            base64=det.base64,
            latitude=det.latitude,
            longitude=det.longitude,
            date_taken=det.date_taken,
            user_id=det.user_id,
            detected_classes=parsed_classes,
            status=det.status,
            detection_points=det.detection_points
        ))
    return response_list

@app.get("/detections/id/{detection_id}", response_model=WasteDetectionResponse)
async def get_detection_by_id(detection_id: str, db: Session = Depends(get_db)):
    detection = db.query(WasteDetection).filter(WasteDetection.id == detection_id).first()
    if detection is None:
        raise HTTPException(status_code=404, detail=f"Detection not found: {detection_id}")
    try:
        parsed_classes = json.loads(detection.detected_classes) if detection.detected_classes else []
    except json.JSONDecodeError:
        parsed_classes = []
    
    return WasteDetectionResponse.from_orm(detection)
  
  
@app.get("/detections/status/{status_value}", response_model=List[WasteDetectionResponse])
async def get_detections_by_status(status_value: str, skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    if status_value not in ["A coletar", "Recusada"]:
        raise HTTPException(status_code=400, detail="Invalid status: 'A coletar' or 'Recusada'.")
    detections = db.query(WasteDetection).filter(WasteDetection.status == status_value).offset(skip).limit(limit).all()
    response_list = []
    for det in detections:
        try:
            parsed_classes = json.loads(det.detected_classes) if det.detected_classes else []
        except json.JSONDecodeError:
            parsed_classes = []
        response_list.append(WasteDetectionResponse(
            id=det.id,
            base64=det.base64,
            latitude=det.latitude,
            longitude=det.longitude,
            date_taken=det.date_taken,
            user_id=det.user_id,
            detected_classes=parsed_classes,
            status=det.status,
            detection_points=det.detection_points
        ))
    return response_list

@app.get("/auth/callback")
async def oauth_callback(request: Request):
    """
    OAuth callback endpoint for Google authentication via Appwrite.
    Returns HTML page that redirects to app deep link using JavaScript.
    """
    try:
        user_id = request.query_params.get('userId')
        secret = request.query_params.get('secret')
        
        if not user_id or not secret:
            deep_link = 'lixoultimate://localhost/?error=missing_credentials'
        else:
            deep_link = f'lixoultimate://localhost/?userId={user_id}&secret={secret}'
        
        # Return HTML page with JavaScript redirect
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Redirecting...</title>
            <meta charset="UTF-8">
        </head>
        <body>
            <h1>Autenticação bem-sucedida!</h1>
            <p>Redirecionando para o aplicativo...</p>
            <script>
                window.location.href = '{deep_link}';
            </script>
        </body>
        </html>
        """
        
        return HTMLResponse(content=html_content, status_code=200)
        
    except Exception as e:
        print(f"OAuth callback error: {e}")
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Error</title>
            <meta charset="UTF-8">
        </head>
        <body>
            <h1>Erro na autenticação</h1>
            <p>Ocorreu um erro. Tente novamente.</p>
            <script>
                window.location.href = 'lixoultimate://localhost/?error=callback_failed';
            </script>
        </body>
        </html>
        """
        return HTMLResponse(content=html_content, status_code=200)