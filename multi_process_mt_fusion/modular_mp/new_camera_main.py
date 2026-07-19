# main.py
# Clean FastAPI app for frontend integration
# Handles: camera registration, live MJPEG streams
# Old inference app is kept separately in main_inference.py

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from camera_routes import router as camera_router
from stream_manager import stop_all
#update
from zone_routes import router as zone_router
from inference_routes import router as inference_router

app = FastAPI(title="DOER.AI Camera API")

# ─────────────────────────────────────────
# CORS — allow React frontend
# ─────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # change to your frontend URL in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────────────────────────────────────
# Routers
# ─────────────────────────────────────────
app.include_router(camera_router)
# new update
app.include_router(zone_router)
app.include_router(inference_router)    
# ─────────────────────────────────────────
# Startup / Shutdown
# ─────────────────────────────────────────
@app.on_event("startup")
def on_startup():
    print("[Server] DOER.AI Camera API started")


@app.on_event("shutdown")
def on_shutdown():
    print("[Server] Shutting down — stopping all streams")
    stop_all()


# ─────────────────────────────────────────
# Health check
# ─────────────────────────────────────────
@app.get("/")
def health():
    return {"status": "ok", "message": "DOER.AI Camera API running"}