# zone_routes.py
# zone API endpoint

import cv2
from fastapi import APIRouter, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel
from typing import List

from zone_store import (
    add_zone, get_all_zones,
    get_zones_by_camera,get_zone,
    delete_zone, toggle_zone, 
    delete_zones_by_camera
)

from stream_manager import get_frame

router = APIRouter(prefix="/api")


# -------------------------
# Pydantic models
# -------------------------

class Point(BaseModel):
    x: float # normalized 0-1
    y: float # normalized 0-1

class ZoneCreate(BaseModel):
    cameraId: str
    name: str
    points: List[Point]
    enabled: bool = True


# --------------------------
# Routes
# --------------------------

@router.get("/zones")
def list_all_zones():
    """Return all zones."""
    return get_all_zones()


@router.get("/zones/camera/{camera_id}")
def list_zones_by_camera(camera_id:str):
    """Return all zones for aa specific camera."""
    return get_zones_by_camera(camera_id)

@router.post("/zones",status_code = 201)
def create_zone(body: ZoneCreate):
    """
    save a new zone for a camera
    Points are normalized (0-1) co ordinates.
    """

    if len(body.points) < 3 :
        raise HTTPException(
            status_code=400,
            detail="Zone must at least 3 points"
        )
    
    # convert Point objects to dicts
    points = [{"x": p.x, "y": p.y} for p in body.points]

    zone = add_zone(
        camera_id = body.cameraId,
        name = body.name,
        points = points,
        enabled = body.enabled
    )

    return zone


@router.delete("/zones/{zone_id}")
def remove_zone(zone_id: str):
    """Delete a zone by id."""

    zone = get_zone(zone_id)
    if not zone:
        raise HTTPException(status_code = 404, detail = "Zone not found")
    
    delete_zone(zone_id)
    return {"message": f"Zone {zone_id} deleted successfully"}


@router.patch("/zones/{zone_id}/toggle")
def toggle_zone_status(zone_id: str):
    """Toggle zone enabled/disabled."""
    
    zone = get_zone(zone_id)
    if not zone:
        raise HTTPException(status_code = 404, detail="Zone not found")
    
    updated = toggle_zone(zone_id)
    return updated


@router.get("/cameras/{camera_id}/snapshot")
def get_snapshot(camera_id:str):
    """
    Get a single frame from camera as jpeg image.
    Used as background in frontend zone drawing canvas.
    """

    frame = get_frame(camera_id)

    if frame is None:
        raise HTTPException(
            status_code = 404,
            detail = "No frame avilable. Make sure camera is streaming"
        )
    
    ret, buffer = cv2.imencode(
        ".jpg", frame,
        [cv2.IMWRITE_JPEG_QUALITY,85]
    )

    if not ret:
        raise HTTPException(
            status_code=500,
            detail="Failed to encode frame"
        )

    return Response(
        content=buffer.tobytes(),
        media_type="image/jpeg"
    )