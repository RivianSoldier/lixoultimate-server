from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel
from dotenv import load_dotenv
import json
import os
from typing import List, Optional, Any

from sqlalchemy import create_engine, Column, Float, String, Integer, ForeignKey
from sqlalchemy.orm import sessionmaker, Session, relationship
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.types import JSON

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
        from_attributes = True

class WasteDetectionResponse(BaseModel):
    id: str
    base64: str
    latitude: float
    longitude: float
    date_taken: str
    user_id: str
    detected_classes: List[str]
    status: str
    detection_points: Optional[Any] = None

    class Config:
        from_attributes = True

class WasteDetectionSmallerResponse(BaseModel):
    id: str
    date_taken: str
    user_id: str
    status: str

    class Config:
        from_attributes = True

class ClassCount(BaseModel):
    nome: str
    quantidade: int

class WasteDetectionMapResponse(BaseModel):
    id: str
    lat: float
    lng: float
    foto: str
    classes: List[ClassCount]
    detection_points: Optional[dict] = None

    class Config:
        from_attributes = True

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

@app.get("/detections_achievements/user/{user_id}", response_model=List[WasteDetectionSmallerResponse])
async def get_detections_achievements_by_user_from_db_service(user_id: str, skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    detections = db.query(WasteDetection).filter(WasteDetection.user_id == user_id).offset(skip).limit(limit).all()
    if not detections:
        user = db.query(User).filter(User.id == user_id).first()
        if not user:
            raise HTTPException(status_code=404, detail=f"User not found with id: {user_id}")
        return []
    return detections

@app.get("/detections/map/status/{status_value}", response_model=List[WasteDetectionMapResponse])
async def get_detections_for_map(status_value: str, skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    """
    Get detections formatted for map frontend display.
    Returns detections with class counts and proper field names.
    """
    if status_value not in ["A coletar", "Recusada"]:
        raise HTTPException(status_code=400, detail="Invalid status value. Must be 'A coletar' or 'Recusada'.")
    
    detections = db.query(WasteDetection).filter(WasteDetection.status == status_value).offset(skip).limit(limit).all()
    
    response_list = []
    for det in detections:
        class_counts = {"papel": 0, "plastico": 0, "vidro": 0, "metal": 0}
        
        if det.detection_points and isinstance(det.detection_points, dict):
            if "class_counts" in det.detection_points:
                stored_counts = det.detection_points["class_counts"]
                for class_name in ["papel", "plastico", "vidro", "metal"]:
                    class_counts[class_name] = stored_counts.get(class_name, 0)
        
        classes = [
            ClassCount(nome="papel", quantidade=class_counts["papel"]),
            ClassCount(nome="plastico", quantidade=class_counts["plastico"]),
            ClassCount(nome="vidro", quantidade=class_counts["vidro"]),
            ClassCount(nome="metal", quantidade=class_counts["metal"])
        ]
        
        response_list.append(WasteDetectionMapResponse(
            id=det.id,
            lat=det.latitude,
            lng=det.longitude,
            foto=det.base64,
            classes=classes,
            detection_points=det.detection_points
        ))
    
    return response_list