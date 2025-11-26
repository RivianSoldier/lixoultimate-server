from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.responses import RedirectResponse, StreamingResponse
from pydantic import BaseModel
from dotenv import load_dotenv
import json
import os
from typing import List, Optional, Any
import base64
import io
import csv
import time
from PIL import Image
from sqlalchemy import create_engine, Column, Float, String, Integer, ForeignKey
from sqlalchemy.orm import sessionmaker, Session, relationship
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.types import JSON
from datetime import datetime
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut, GeocoderServiceError


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
    
    # Existing columns
    collected_by = Column(String, ForeignKey("users.id"), nullable=True, index=True)
    collection_date = Column(String, nullable=True)

    # --- ADD THESE TWO LINES ---
    not_found_by = Column(String, ForeignKey("users.id"), nullable=True, index=True)
    not_found_date = Column(String, nullable=True)

Base.metadata.create_all(engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Initialize geocoder for address conversion
geolocator = Nominatim(user_agent="lixoultimate-server")

def get_address_from_coordinates(lat: float, lng: float, timeout: int = 3) -> str:
    """
    Convert latitude and longitude to a readable address.
    Returns formatted address or coordinates if geocoding fails.
    Uses a shorter timeout to prevent worker timeouts.
    """
    try:
        location = geolocator.reverse(f"{lat}, {lng}", timeout=timeout, language='pt')
        if location and location.address:
            return location.address
        return f"{lat}, {lng}"
    except (GeocoderTimedOut, GeocoderServiceError, Exception) as e:
        print(f"Geocoding error for {lat}, {lng}: {e}")
        return f"{lat}, {lng}"

def format_classes_for_csv(detection_points: dict) -> str:
    """
    Extract and format class counts from detection_points for CSV export.
    Returns formatted string like "Papel: 5, Plástico: 3"
    """
    if not detection_points or not isinstance(detection_points, dict):
        return "Lixo: 1"
    
    class_counts = {"papel": 0, "plastico": 0, "vidro": 0, "metal": 0}
    
    if "class_counts" in detection_points:
        stored_counts = detection_points["class_counts"]
        for class_name in ["papel", "plastico", "vidro", "metal"]:
            class_counts[class_name] = stored_counts.get(class_name, 0)
    
    # Build formatted string
    class_name_map = {
        "papel": "Papel",
        "plastico": "Plástico",
        "vidro": "Vidro",
        "metal": "Metal"
    }
    
    parts = []
    for class_name, count in class_counts.items():
        if count > 0:
            parts.append(f"{class_name_map[class_name]}: {count}")
    
    return ", ".join(parts) if parts else "Lixo: 1"

def compress_base64_image(base64_str: str, quality: int = 60, max_size: tuple = (1024, 1024)) -> str:
    """
    Compress a base64 encoded image.
    
    Args:
        base64_str: Base64 encoded image string (with or without data URI prefix)
        quality: JPEG quality (1-100, lower = smaller file)
        max_size: Maximum dimensions (width, height) to resize to
    
    Returns:
        Compressed base64 string with data URI prefix
    """
    try:
        if base64_str.startswith('data:image'):
            header, base64_data = base64_str.split(',', 1)
        else:
            base64_data = base64_str
            header = None
        
        img_data = base64.b64decode(base64_data)
        img = Image.open(io.BytesIO(img_data))
        
        if img.mode in ('RGBA', 'LA', 'P'):
            background = Image.new('RGB', img.size, (255, 255, 255))
            if img.mode == 'P':
                img = img.convert('RGBA')
            background.paste(img, mask=img.split()[-1] if img.mode in ('RGBA', 'LA') else None)
            img = background
        elif img.mode != 'RGB':
            img = img.convert('RGB')
        
        if img.width > max_size[0] or img.height > max_size[1]:
            img.thumbnail(max_size, Image.LANCZOS)
        
        buffer = io.BytesIO()
        img.save(buffer, format='JPEG', quality=quality, optimize=True)
        buffer.seek(0)
        
        compressed_base64 = base64.b64encode(buffer.read()).decode('utf-8')
        
        # Return just the base64 string without prefix (frontend will add it if needed)
        return compressed_base64
    
    except Exception as e:
        print(f"Image compression failed: {e}")
        # Return original format on error - strip prefix if present
        if base64_str.startswith('data:image'):
            return base64_str.split(',', 1)[1]
        return base64_str

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
    collected_by: Optional[str] = None
    collection_date: Optional[str] = None

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

class CollectionRequest(BaseModel):
    collector_user_id: str

class CollectionResponse(BaseModel):
    success: bool
    detection_id: str
    status: str
    collected_by: Optional[str] = None
    collection_date: Optional[str] = None
    not_found_by: Optional[str] = None
    not_found_date: Optional[str] = None
    message: str

class WasteDetectionMapResponse(BaseModel):
    id: str
    lat: float
    lng: float
    foto: str
    classes: List[ClassCount]
    detection_points: Optional[Any] = None
    date: str  # Campo de data adicionado para o frontend

    class Config:
        from_attributes = True
        
class CollectorActivityResponse(BaseModel):
    id: str
    foto: str
    classes: List[ClassCount]
    detection_points: Optional[Any] = None
    lat: float
    lng: float
    date: str
    status: str
    dataColetado: Optional[str] = None

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
        
        compressed_image = compress_base64_image(det.base64, quality=70, max_size=(1024, 1024))
        
        response_list.append(WasteDetectionResponse(
            id=det.id,
            base64=compressed_image,
            latitude=det.latitude,
            longitude=det.longitude,
            date_taken=det.date_taken,
            user_id=det.user_id,
            detected_classes=parsed_classes,
            status=det.status,
            detection_points=det.detection_points,
            collected_by=det.collected_by,
            collection_date=det.collection_date
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
        
        compressed_image = compress_base64_image(det.base64, quality=70, max_size=(1024, 1024))
        
        response_list.append(WasteDetectionResponse(
            id=det.id,
            base64=compressed_image,
            latitude=det.latitude,
            longitude=det.longitude,
            date_taken=det.date_taken,
            user_id=det.user_id,
            detected_classes=parsed_classes,
            status=det.status,
            detection_points=det.detection_points,
            collected_by=det.collected_by,
            collection_date=det.collection_date
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
    
    compressed_image = compress_base64_image(detection.base64, quality=75, max_size=(1280, 1280))

    return WasteDetectionResponse(
        id=detection.id,
        base64=compressed_image,
        latitude=detection.latitude,
        longitude=detection.longitude,
        date_taken=detection.date_taken,
        user_id=detection.user_id,
        detected_classes=parsed_classes,
        status=detection.status,
        detection_points=detection.detection_points,
        collected_by=detection.collected_by,
        collection_date=detection.collection_date
    )

@app.get("/detections/status/{status_value}", response_model=List[WasteDetectionResponse])
async def get_detections_by_status_from_db_service(status_value: str, skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    if status_value not in ["A coletar", "Recusada", "Coletado"]:
        raise HTTPException(status_code=400, detail="Invalid status value. Must be 'A coletar', 'Recusada', or 'Coletado'.")
    detections = db.query(WasteDetection).filter(WasteDetection.status == status_value).offset(skip).limit(limit).all()
    response_list = []
    for det in detections:
        try:
            parsed_classes = json.loads(det.detected_classes) if det.detected_classes else []
        except json.JSONDecodeError:
            parsed_classes = []
        
        compressed_image = compress_base64_image(det.base64, quality=70, max_size=(1024, 1024))
        
        response_list.append(WasteDetectionResponse(
            id=det.id,
            base64=compressed_image,
            latitude=det.latitude,
            longitude=det.longitude,
            date_taken=det.date_taken,
            user_id=det.user_id,
            detected_classes=parsed_classes,
            status=det.status,
            detection_points=det.detection_points,
            collected_by=det.collected_by,
            collection_date=det.collection_date
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
        
        compressed_image = compress_base64_image(det.base64, quality=50, max_size=(800, 800))
        
        response_list.append(WasteDetectionMapResponse(
            id=det.id,
            lat=det.latitude,
            lng=det.longitude,
            foto=compressed_image,
            classes=classes,
            detection_points=det.detection_points,
            date=det.date_taken  # Adicionando o campo de data da detecção
        ))
    
    return response_list

@app.post("/detections/{detection_id}/collect", response_model=CollectionResponse)
async def collect_waste(detection_id: str, request: CollectionRequest, db: Session = Depends(get_db)):
    """
    Mark a waste detection as collected.
    Changes status to 'Coletado' and records collector info.
    This action is permanent and cannot be undone.
    Collector does not need to be a registered user.
    """
    # Verify detection exists
    detection = db.query(WasteDetection).filter(WasteDetection.id == detection_id).first()
    if not detection:
        raise HTTPException(status_code=404, detail=f"Detection not found: {detection_id}")
    
    # Verify status is "A coletar"
    if detection.status != "A coletar":
        raise HTTPException(
            status_code=400, 
            detail=f"Cannot collect detection with status '{detection.status}'. Only 'A coletar' detections can be collected."
        )
    
    # Update detection status and collection info
    detection.status = "Coletado"
    detection.collected_by = request.collector_user_id
    detection.collection_date = datetime.now().isoformat()
    
    try:
        db.commit()
        db.refresh(detection)
        print(f"✅ Detection {detection_id} collected by {request.collector_user_id}")
    except Exception as e:
        db.rollback()
        print(f"!!! DATABASE ERROR during collection: {type(e).__name__} - {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to record collection: {str(e)}")
    
    return CollectionResponse(
        success=True,
        detection_id=detection.id,
        status=detection.status,
        collected_by=detection.collected_by,
        collection_date=detection.collection_date,
        message="Waste successfully marked as collected"
    )
    
@app.post("/detections/{detection_id}/not_found", response_model=CollectionResponse)
async def mark_detection_not_found(detection_id: str, request: CollectionRequest, db: Session = Depends(get_db)):
    """
    Mark a waste detection as not found.
    Changes status to 'Não encontrado' and records user info.
    This action is permanent and cannot be undone.
    """
    # Verify detection exists
    detection = db.query(WasteDetection).filter(WasteDetection.id == detection_id).first()
    if not detection:
        raise HTTPException(status_code=404, detail=f"Detection not found: {detection_id}")

    # Verify status is "A coletar"
    if detection.status != "A coletar":
        raise HTTPException(
            status_code=400,
            detail=f"Cannot mark detection as not found with status '{detection.status}'. Only 'A coletar' detections can be marked as not found."
        )

    # Update detection status and not found info
    detection.status = "Não encontrado"
    detection.not_found_by = request.collector_user_id
    detection.not_found_date = datetime.now().isoformat()

    try:
        db.commit()
        db.refresh(detection)
        print(f"✅ Detection {detection_id} marked as not found by {request.collector_user_id}")
    except Exception as e:
        db.rollback()
        print(f"!!! DATABASE ERROR during not found marking: {type(e).__name__} - {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to record not found: {str(e)}")

    return CollectionResponse(
        success=True,
        detection_id=detection.id,
        status=detection.status,
        not_found_by=detection.not_found_by,
        not_found_date=detection.not_found_date,
        message="Waste successfully marked as not found"
    )

@app.get("/collections/user/{user_id}", response_model=List[WasteDetectionResponse])
async def get_user_collections(user_id: str, skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    """
    Get all waste detections collected by a specific user.
    Returns personalized collection history.
    """
    # Verify user exists
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail=f"User not found: {user_id}")
    
    # Get all detections collected by this user
    detections = db.query(WasteDetection).filter(
        WasteDetection.collected_by == user_id
    ).offset(skip).limit(limit).all()
    
    response_list = []
    for det in detections:
        try:
            parsed_classes = json.loads(det.detected_classes) if det.detected_classes else []
        except json.JSONDecodeError:
            parsed_classes = []
        
        compressed_image = compress_base64_image(det.base64, quality=70, max_size=(1024, 1024))
        
        response_list.append(WasteDetectionResponse(
            id=det.id,
            base64=compressed_image,
            latitude=det.latitude,
            longitude=det.longitude,
            date_taken=det.date_taken,
            user_id=det.user_id,
            detected_classes=parsed_classes,
            status=det.status,
            detection_points=det.detection_points,
            collected_by=det.collected_by,
            collection_date=det.collection_date
        ))
    
    return response_list

@app.get("/detections/available", response_model=List[WasteDetectionResponse])
async def get_available_detections(skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    """
    Get all waste detections available for collection.
    Returns only detections with status 'A coletar' (not yet collected).
    """
    detections = db.query(WasteDetection).filter(
        WasteDetection.status == "A coletar"
    ).offset(skip).limit(limit).all()
    
    response_list = []
    for det in detections:
        try:
            parsed_classes = json.loads(det.detected_classes) if det.detected_classes else []
        except json.JSONDecodeError:
            parsed_classes = []
        
        compressed_image = compress_base64_image(det.base64, quality=70, max_size=(1024, 1024))
        
        response_list.append(WasteDetectionResponse(
            id=det.id,
            base64=compressed_image,
            latitude=det.latitude,
            longitude=det.longitude,
            date_taken=det.date_taken,
            user_id=det.user_id,
            detected_classes=parsed_classes,
            status=det.status,
            detection_points=det.detection_points,
            collected_by=det.collected_by,
            collection_date=det.collection_date
        ))
    
    return response_list

@app.get("/collector/activity/{collector_id}", response_model=List[CollectorActivityResponse])
async def get_collector_activity(collector_id: str, skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    """
    Get all waste detections that a collector has either collected or marked as not found.
    Returns activity history with simplified format for frontend display.
    """
    from sqlalchemy import or_
    
    # Get all detections where this person was the collector or marked as not found
    detections = db.query(WasteDetection).filter(
        or_(
            WasteDetection.collected_by == collector_id,
            WasteDetection.not_found_by == collector_id
        )
    ).offset(skip).limit(limit).all()
    
    response_list = []
    for det in detections:
        # Extract class counts from detection_points
        class_counts = {"papel": 0, "plastico": 0, "vidro": 0, "metal": 0}
        
        if det.detection_points and isinstance(det.detection_points, dict):
            if "class_counts" in det.detection_points:
                stored_counts = det.detection_points["class_counts"]
                for class_name in ["papel", "plastico", "vidro", "metal"]:
                    class_counts[class_name] = stored_counts.get(class_name, 0)
        
        # Build classes list (only include classes with count > 0)
        classes = []
        class_name_map = {
            "papel": "Papel",
            "plastico": "Plástico",
            "vidro": "Vidro",
            "metal": "Metal"
        }
        for class_name, count in class_counts.items():
            if count > 0:
                classes.append(ClassCount(
                    nome=class_name_map[class_name],
                    quantidade=count
                ))
        
        # If no classes detected, add a generic "Lixo" entry
        if not classes:
            classes.append(ClassCount(nome="Lixo", quantidade=1))
        
        # Compress image for frontend
        compressed_image = compress_base64_image(det.base64, quality=50, max_size=(800, 800))
        
        # Determine the action date (collection_date or not_found_date)
        data_coletado = None
        if det.status == "Coletado" and det.collection_date:
            data_coletado = det.collection_date
        elif det.status == "Não encontrado" and det.not_found_date:
            data_coletado = det.not_found_date
        
        response_list.append(CollectorActivityResponse(
            id=det.id,
            foto=compressed_image,
            classes=classes,
            detection_points=det.detection_points,
            lat=det.latitude,
            lng=det.longitude,
            date=det.date_taken,
            status=det.status,
            dataColetado=data_coletado
        ))
    
    return response_list

@app.get("/collections/user/{user_id}/export")
async def export_user_collections(user_id: str, db: Session = Depends(get_db)):
    """
    Export user's collection history as CSV file.
    Returns CSV with columns: Data, Hora, Endereço, Classes, Status, Latitude, Longitude.
    All text in Portuguese for user convenience.
    Note: user_id can be any collector ID, doesn't need to be a registered user.
    """
    # Get all detections collected or marked as not found by this user
    from sqlalchemy import or_
    detections = db.query(WasteDetection).filter(
        or_(
            WasteDetection.collected_by == user_id,
            WasteDetection.not_found_by == user_id
        )
    ).all()
    
    # Create CSV in memory with UTF-8 BOM
    output = io.StringIO()
    output.write('\ufeff')  # UTF-8 BOM for Excel compatibility
    writer = csv.writer(output, delimiter=';')
    
    # Write header
    writer.writerow(["Data Detecção", "Data Status", "Hora Status", "Endereço", "Classes", "Status", "Latitude", "Longitude"])
    
    # Write data rows
    for det in detections:
        # Determine the action date (status change date)
        action_date = None
        if det.status == "Coletado" and det.collection_date:
            action_date = det.collection_date
        elif det.status == "Não encontrado" and det.not_found_date:
            action_date = det.not_found_date
        
        if action_date:
            try:
                # Parse ISO datetime
                dt = datetime.fromisoformat(action_date)
                status_date_str = dt.strftime("%d/%m/%Y")
                status_time_str = dt.strftime("%H:%M:%S")
            except:
                status_date_str = action_date.split("T")[0] if "T" in action_date else action_date
                status_time_str = action_date.split("T")[1].split(".")[0] if "T" in action_date else "--:--:--"
        else:
            status_date_str = "--/--/----"
            status_time_str = "--:--:--"
        
        # Format detection date
        try:
            detection_dt = datetime.fromisoformat(det.date_taken)
            detection_date_str = detection_dt.strftime("%d/%m/%Y")
        except:
            detection_date_str = det.date_taken.split("T")[0] if "T" in det.date_taken else det.date_taken
        
        # Get address from coordinates
        address = get_address_from_coordinates(det.latitude, det.longitude)
        time.sleep(1)  # Respect Nominatim's rate limit (1 request per second)
        
        # Format classes
        classes_str = format_classes_for_csv(det.detection_points)
        
        # Write row
        writer.writerow([
            detection_date_str,
            status_date_str,
            status_time_str,
            address,
            classes_str,
            det.status,
            f"{det.latitude:.6f}",
            f"{det.longitude:.6f}"
        ])
    
    # Prepare response
    output.seek(0)
    filename = f"historico_coletas_{user_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    
    return StreamingResponse(
        iter([output.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename={filename}"}
    )

@app.get("/collections/export")
async def export_all_collections(db: Session = Depends(get_db)):
    """
    Export all collections history as anonymized CSV file.
    Returns CSV without user identification for data protection.
    Columns: Data, Hora, Endereço, Classes, Status, Latitude, Longitude.
    """
    # Get all detections that have been collected or marked as not found
    detections = db.query(WasteDetection).filter(
        WasteDetection.status.in_(["Coletado", "Não encontrado"])
    ).all()
    
    # Create CSV in memory with UTF-8 BOM
    output = io.StringIO()
    output.write('\ufeff')  # UTF-8 BOM for Excel compatibility
    writer = csv.writer(output, delimiter=';')
    
    # Write header
    writer.writerow(["Data Detecção", "Data Status", "Hora Status", "Endereço", "Classes", "Status", "Latitude", "Longitude"])
    
    # Write data rows
    for det in detections:
        # Determine the action date (status change date)
        action_date = None
        if det.status == "Coletado" and det.collection_date:
            action_date = det.collection_date
        elif det.status == "Não encontrado" and det.not_found_date:
            action_date = det.not_found_date
        
        if action_date:
            try:
                # Parse ISO datetime
                dt = datetime.fromisoformat(action_date)
                status_date_str = dt.strftime("%d/%m/%Y")
                status_time_str = dt.strftime("%H:%M:%S")
            except:
                status_date_str = action_date.split("T")[0] if "T" in action_date else action_date
                status_time_str = action_date.split("T")[1].split(".")[0] if "T" in action_date else "--:--:--"
        else:
            status_date_str = "--/--/----"
            status_time_str = "--:--:--"
        
        # Format detection date
        try:
            detection_dt = datetime.fromisoformat(det.date_taken)
            detection_date_str = detection_dt.strftime("%d/%m/%Y")
        except:
            detection_date_str = det.date_taken.split("T")[0] if "T" in det.date_taken else det.date_taken
        
        # Get address from coordinates
        address = get_address_from_coordinates(det.latitude, det.longitude)
        time.sleep(1)  # Respect Nominatim's rate limit (1 request per second)
        
        # Format classes
        classes_str = format_classes_for_csv(det.detection_points)
        
        # Write row (no user identification)
        writer.writerow([
            detection_date_str,
            status_date_str,
            status_time_str,
            address,
            classes_str,
            det.status,
            f"{det.latitude:.6f}",
            f"{det.longitude:.6f}"
        ])
    
    # Prepare response
    output.seek(0)
    filename = f"historico_coletas_global_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    
    return StreamingResponse(
        iter([output.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename={filename}"}
    )