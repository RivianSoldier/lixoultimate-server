from datetime import date
from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from pydantic import BaseModel
from dotenv import load_dotenv
import json
import base64
import os
from PIL import Image
from typing import List, Optional

import io
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


load_dotenv()
app = FastAPI()

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
  detections = relationship("WasteDetection", back_populates="owner")

class WasteDetection(Base):
  __tablename__ = "waste_detections"
  id = Column(String, primary_key=True, index=True)
  base64 = Column(String)
  latitude = Column(Float)
  longitude = Column(Float)
  date_taken = Column(String)
  user_id = Column(String, ForeignKey("users.id"), index=True)
  owner = relationship("User", back_populates="detections")
  detected_classes = Column(String)
  status = Column(String, index=True)

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

  class Config:
    orm_mode = True


safe_globals_list = [
  DetectionModel, torch.nn.Sequential, Conv, Concat, C3k2, C3, SPPF, Bottleneck, C2f, Detect, torch.nn.Conv2d, torch.nn.BatchNorm2d, torch.nn.ReLU, torch.nn.LeakyReLU, torch.nn.SiLU, torch.nn.Sigmoid, torch.nn.Hardswish, torch.nn.Upsample, torch.nn.MaxPool2d, torch.nn.AdaptiveAvgPool2d, torch.nn.AdaptiveMaxPool2d
  ]

torch.serialization.add_safe_globals(safe_globals_list)

model = YOLO("best.pt")

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
	if model is None:
		raise HTTPException(status_code=503, detail="Model not loaded. Check service logs.")

	print("Received data for classification:", json.dumps(data.model_dump(), indent=2))
	
	try:
		img_data = base64.b64decode(data.base64)
		img = Image.open(io.BytesIO(img_data)).convert("RGB")
	except Exception as e:
		raise HTTPException(status_code=400, detail=f"Invalid base64 image data: {e}")
	
	try:
		results = model(img)
	except Exception as e:
		raise HTTPException(status_code=500, detail=f"Error during image processing: {e}")

	detected_classes_names = []
	if results and results[0].boxes and results[0].boxes.cls is not None:
		for res_box in results[0].boxes:
			if res_box.cls is not None:
				for cls_tensor in res_box.cls:
					cls_index = int(cls_tensor.item())
					if 0 <= cls_index < len(results[0].names):
						detected_classes_names.append(results[0].names[cls_index])
					else:
						print(f"Warning: class index {cls_index} out of bounds for names list.")
	
	current_status = "Recusada"
	coins_to_award = 0
	
	user = db.query(User).filter(User.id == data.userId).first()
	if not user:
		print(f"Usuário {data.userId} nao encontrado, criando novo usuario.")
		user = User(id=data.userId, coins=0)
		db.add(user)
		try:
			db.commit() 
			db.refresh(user)
			print(f"novo usuario {user.id} criado.")
		except Exception as e: 
			db.rollback()
			print(f"!!! DATABASE ERROR ON CREATING USER: {type(e).__name__} - {str(e)}")
			traceback.print_exc()
			user = db.query(User).filter(User.id == data.userId).first()
			if not user:
				raise HTTPException(status_code=500, detail=f"Failed to create or find user after rollback: {str(e)}")
	else:
		print(f"User {user.id} found with {user.coins} coins.")

	if "lixo" in detected_classes_names:
		current_status = "A coletar"
		coins_to_award = 100
		print(f"Lixo detectado. Recompensa de {coins_to_award} moedas para: {user.id}. Moedas atuais: {user.coins}")
		user.coins += coins_to_award 
		print(f"Moedas do usuário {user.id} atualizadas para: {user.coins}")
	if not 'lixo' in detected_classes_names and user.coins >= 50:
		current_status = "Recusada"
		print(f"Nenhum 'lixo' detectado. Removendo 50 moedas de {user.id}. Moedas atuais: {user.coins}")
		user.coins = max(0, user.coins - 50)
	if not 'lixo' in detected_classes_names and user.coins < 50:
		current_status = "Recusada"
		print(f"Nenhum 'lixo' detectado e moedas insuficientes ({user.coins}). Definindo moedas para 0.")
		user.coins = 0
	
	new_detection_id = str(uuid.uuid4())
	try:
		print(f"Attempting to save detection {new_detection_id} for user {user.id} with status '{current_status}'")
		detection_db_entry = WasteDetection(
			id=new_detection_id,
			base64=data.base64,
			latitude=data.latitude,
			longitude=data.longitude,
			date_taken=data.dateTaken,
			user_id=user.id, 
			detected_classes=json.dumps(detected_classes_names),
			status=current_status,
		)
		db.add(detection_db_entry)
		
		db.commit()
		print(f"Successfully committed detection {new_detection_id} and user updates for {user.id}.")
		db.refresh(detection_db_entry)
		db.refresh(user)

	except Exception as e:
		db.rollback()
		print(f"!!! DATABASE ERROR ON SAVING DETECTION/UPDATING COINS FOR USER {user.id}: {type(e).__name__} - {str(e)}")
		traceback.print_exc() 
		raise HTTPException(status_code=500, detail=f"Database error on saving detection/coins: {str(e)}")
	
	return {
		"id": new_detection_id,
		"status": "success",
		"message": "Image classified and data saved",
		"detected_classes": detected_classes_names,
		"classification_status": current_status,
		"coins_awarded_this_time": coins_to_award,
		"user_total_coins": user.coins 
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
		except json.JSONDecodeError: parsed_classes = []
		response_list.append(WasteDetectionResponse(
			id=det.id,
   		base64=det.base64,
			latitude=det.latitude,
			longitude=det.longitude,
			date_taken=det.date_taken,
			user_id=det.user_id,
			detected_classes=parsed_classes,
			status=det.status
		))
	return response_list

@app.get("/detections/user/{user_id}", response_model=List[WasteDetectionResponse])
async def get_detections_by_user(user_id: str, skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
	detections = db.query(WasteDetection).filter(WasteDetection.user_id == user_id).offset(skip).limit(limit).all()
	if not detections: # Return empty list instead of 404 if user exists but has no detections
		user = db.query(User).filter(User.id == user_id).first()
		if not user:
			raise HTTPException(status_code=404, detail=f"User not found with id: {user_id}")
		return []
			
	response_list = []
	for det in detections:
		try:
			parsed_classes = json.loads(det.detected_classes) if det.detected_classes else []
		except json.JSONDecodeError: parsed_classes = []
		response_list.append(WasteDetectionResponse(
			id=det.id, latitude=det.latitude, longitude=det.longitude,
			date_taken=det.date_taken, user_id=det.user_id,
			detected_classes=parsed_classes, status=det.status
		))
	return response_list

@app.get("/detections/id/{detection_id}", response_model=WasteDetectionResponse)
async def get_detection_by_id(detection_id: str, db: Session = Depends(get_db)):
	detection = db.query(WasteDetection).filter(WasteDetection.id == detection_id).first()
	if detection is None:
		raise HTTPException(status_code=404, detail=f"Detection not found: {detection_id}")
	try:
		parsed_classes = json.loads(detection.detected_classes) if detection.detected_classes else []
	except json.JSONDecodeError: parsed_classes = []
	return WasteDetectionResponse(
		id=detection.id, latitude=detection.latitude, longitude=detection.longitude,
		date_taken=detection.date_taken, user_id=detection.user_id,
		detected_classes=parsed_classes, status=detection.status
	)

@app.get("/detections/status/{status_value}", response_model=List[WasteDetectionResponse])
async def get_detections_by_status(status_value: str, skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
	if status_value not in ["A coletar", "Recusada"]:
			raise HTTPException(status_code=400, detail="Invalid status: 'A coletar' or 'Recusada'.")
	detections = db.query(WasteDetection).filter(WasteDetection.status == status_value).offset(skip).limit(limit).all()
	response_list = []
	for det in detections:
		try:
			parsed_classes = json.loads(det.detected_classes) if det.detected_classes else []
		except json.JSONDecodeError: parsed_classes = []
		response_list.append(WasteDetectionResponse(
			id=det.id, latitude=det.latitude, longitude=det.longitude,
			date_taken=det.date_taken, user_id=det.user_id,
			detected_classes=parsed_classes, status=det.status
		))
	return response_list