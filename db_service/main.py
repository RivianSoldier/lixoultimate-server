from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel
from dotenv import load_dotenv
import json
import os
from typing import List, Optional, Any

from sqlalchemy import create_engine, Column, Float, String, Integer, ForeignKey, JSON
from sqlalchemy.orm import sessionmaker, Session, relationship
from sqlalchemy.ext.declarative import declarative_base

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
  detection_points = Column(JSON, nullable=True)



Base.metadata.create_all(engine)

def get_db():
	db = SessionLocal()
	try:
		yield db
	finally:
		db.close()

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
		orm_mode = True
        
class WasteDetectionSmallerResponse(BaseModel):
	id: str
	date_taken: str
	user_id: str
	status: str

	class Config:
		orm_mode = True
        
@app.get("/users/", response_model=List[UserResponse])
async def get_all_users_from_db_service(skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
	users = db.query(User).offset(skip).limit(limit).all()
	return users

@app.get("/users/{user_id}", response_model=UserResponse)
async def get_user_by_id_from_db_service(user_id: str, db: Session = Depends(get_db)):
	user = db.query(User).filter(User.id == user_id).first()
	if user is None:
		raise HTTPException(status_code=404, detail=f"User not found with id: {user_id}")
	return user

@app.get("/detections/", response_model=List[WasteDetectionResponse])
async def get_all_detections_from_db_service(skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
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
async def get_detections_by_user_from_db_service(user_id: str, skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
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
async def get_detection_by_id_from_db_service(detection_id: str, db: Session = Depends(get_db)):
	detection = db.query(WasteDetection).filter(WasteDetection.id == detection_id).first()
	if detection is None:
		raise HTTPException(status_code=404, detail=f"Detection not found: {detection_id}")
	try:
		parsed_classes = json.loads(detection.detected_classes) if detection.detected_classes else []
	except json.JSONDecodeError:
		parsed_classes = []
	return WasteDetectionResponse(
		id=detection.id,
		base64=detection.base64,
		latitude=detection.latitude,
		longitude=detection.longitude,
		date_taken=detection.date_taken,
		user_id=detection.user_id,
		detected_classes=parsed_classes,
		status=detection.status,
		detection_points=detection.detection_points
	)

@app.get("/detections/status/{status_value}", response_model=List[WasteDetectionResponse])
async def get_detections_by_status_from_db_service(status_value: str, skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
	if status_value not in ["A coletar", "Recusada"]:
		raise HTTPException(status_code=400, detail="Invalid status value. Must be 'A coletar' or 'Recusada'.")
	detections = db.query(WasteDetection).filter(WasteDetection.status == status_value).offset(skip).limit(limit).all()
	response_list = []
	for det in detections:
		try:
			parsed_classes = json.loads(det.detected_classes) if det.detected_classes else []
		except json.JSONDecodeError:
			parsed_classes = []
		response_list.append(WasteDetectionResponse(
			id=det.id,
			latitude=det.latitude,
			longitude=det.longitude,
			date_taken=det.date_taken,
			user_id=det.user_id,
			detected_classes=parsed_classes,
			status=det.status,
			detection_points=det.detection_points
		))
	return response_list

@app.get("/detections_achievements/user/{user_id}", response_model=List[WasteDetectionSmallerResponse])
async def get_detections_achievements_by_user_from_db_service(user_id: str, skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
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
		response_list.append(WasteDetectionSmallerResponse(
			id=det.id,
			date_taken=det.date_taken,
			user_id=det.user_id,
			status=det.status
		))
	return response_list
