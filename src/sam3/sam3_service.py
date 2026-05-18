"""
SAM3 Segmentation Microservice

FastAPI service for player segmentation using SAM3

Run in sam3_env environment:
    uvicorn sam3_service:app --host 0.0.0.0 --port 8001

Architecture:
    Flask (SoccerNet env, port 5000) → HTTP → SAM3 Service (sam3_env, port 8001)
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import torch
import numpy as np
import cv2
from pathlib import Path
import logging
from contextlib import asynccontextmanager

# SAM3 imports (only available in sam3_env)
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor
from PIL import Image

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global model instances (loaded once at startup)
sam3_model = None
sam3_processor = None

# Paths
SAM3_SERVICE_DIR = Path(__file__).resolve().parent
BPE_PATH = SAM3_SERVICE_DIR / "assets" / "bpe_simple_vocab_16e6.txt.gz"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager - loads SAM3 model on startup, cleans up on shutdown.
    """
    global sam3_model, sam3_processor
    
    logger.info("=" * 60)
    logger.info("SAM3 Segmentation Service Starting...")
    logger.info("=" * 60)
    
    try:
        # Load SAM3 model once at startup
        logger.info("Loading SAM3 model from HuggingFace...")

        if not BPE_PATH.exists():
            raise FileNotFoundError(f"SAM3 BPE tokenizer file not found: {BPE_PATH}")

        logger.info(f"Using SAM3 BPE tokenizer: {BPE_PATH}")

        sam3_model = build_sam3_image_model(
            device="cuda" if torch.cuda.is_available() else "cpu",
            load_from_HF=True,
            enable_segmentation=True,
            enable_inst_interactivity=True,
            bpe_path=str(BPE_PATH)
        )
        
        # Create processor
        sam3_processor = Sam3Processor(
            sam3_model, 
            device="cuda" if torch.cuda.is_available() else "cpu"
        )
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"✓ SAM3 loaded successfully on {device}")
        logger.info("=" * 60)
        logger.info("Service ready to accept requests on port 8001")
        logger.info("=" * 60)
        
    except Exception as e:
        logger.error(f"Failed to load SAM3 model: {e}")
        raise
    
    yield
    
    logger.info("Shutting down SAM3 service...")
    sam3_model = None
    sam3_processor = None


# Initialize FastAPI with lifespan
app = FastAPI(
    title="SAM3 Segmentation Service",
    description="Microservice for SAM3 player segmentation",
    version="1.0.0",
    lifespan=lifespan
)


# Request/Response models
class SegmentationRequest(BaseModel):
    """Request schema for segmentation endpoint."""
    image_path: str
    prompt: str = "soccer player"
    confidence_threshold: float = 0.5
    output_dir: Optional[str] = None


class SegmentationResponse(BaseModel):
    """Response schema for segmentation endpoint."""
    success: bool
    mask_path: Optional[str] = None
    preview_path: Optional[str] = None
    num_instances: int = 0
    error: Optional[str] = None


@app.get("/")
async def root():
    """Health check endpoint."""
    return {
        "service": "SAM3 Segmentation Service",
        "status": "running",
        "model_loaded": sam3_model is not None,
        "device": "cuda" if torch.cuda.is_available() else "cpu"
    }


@app.get("/health")
async def health_check():
    """Detailed health check."""
    if sam3_model is None:
        raise HTTPException(status_code=503, detail="SAM3 model not loaded")
    
    return {
        "status": "healthy",
        "model": "SAM3",
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "ready": True
    }


@app.post("/segment", response_model=SegmentationResponse)
async def segment_players(request: SegmentationRequest):
    """
    Segment players using SAM3 text prompts.
    
    Args:
        request: SegmentationRequest with image_path and prompt
    
    Returns:
        SegmentationResponse with mask_path and metadata
    """
    try:
        # Validate model is loaded
        if sam3_model is None or sam3_processor is None:
            raise HTTPException(
                status_code=503,
                detail="SAM3 model not loaded. Service may still be starting."
            )
        
        # Validate image exists
        image_path = Path(request.image_path)
        if not image_path.exists():
            raise HTTPException(
                status_code=404,
                detail=f"Image not found: {request.image_path}"
            )
        
        logger.info(f"Segmenting: {image_path}")
        logger.info(f"Prompt: '{request.prompt}'")
        
        # Load image
        image_pil = Image.open(image_path).convert("RGB")
        w, h = image_pil.size
        
        # Set image for SAM3
        inference_state = sam3_processor.set_image(image_pil)
        
        # Segment with text prompt
        output = sam3_processor.set_text_prompt(
            state=inference_state,
            prompt=request.prompt
        )
        
        # Extract results
        masks = output["masks"]  # [num_instances, H, W]
        scores = output["scores"]  # [num_instances]
        
        logger.info(f"Found {len(masks)} instance(s)")
        
        # Filter by confidence
        valid_masks = []
        for i, (mask, score) in enumerate(zip(masks, scores)):
            if torch.is_tensor(score):
                score_val = float(score.detach().to(torch.float32).cpu().item())
            else:
                score_val = float(score)

            if score_val >= request.confidence_threshold:
                mask_np = mask.cpu().numpy() if torch.is_tensor(mask) else mask
                valid_masks.append(mask_np)
                logger.info(f"  Instance {i}: score={score_val:.3f} ✓")
            else:
                logger.info(f"  Instance {i}: score={score_val:.3f} ✗ (filtered)")
        
        if not valid_masks:
            logger.warning("No instances passed confidence threshold")
            return SegmentationResponse(
                success=True,
                num_instances=0,
                error="No objects detected with sufficient confidence"
            )
        
        # Create composite mask (combine all valid masks)
        composite_mask = np.zeros((h, w), dtype=bool)
        for mask in valid_masks:
            composite_mask = np.logical_or(composite_mask, mask > 0)
        
        # Refine mask
        refined_mask = refine_mask((composite_mask * 255).astype(np.uint8))
        
        # Determine output directory
        if request.output_dir:
            output_dir = Path(request.output_dir)
        else:
            output_dir = image_path.parent.parent / "masks"
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate output filenames
        base_name = image_path.stem
        mask_filename = f"{base_name}_mask.png"
        preview_filename = f"{base_name}_preview.png"
        
        mask_path = output_dir / mask_filename
        preview_path = output_dir / preview_filename
        
        # Save mask
        cv2.imwrite(str(mask_path), refined_mask)
        logger.info(f"Saved mask: {mask_path}")
        
        # Create preview (image with mask applied)
        create_preview(image_path, refined_mask, preview_path)
        logger.info(f"Saved preview: {preview_path}")
        
        return SegmentationResponse(
            success=True,
            mask_path=str(mask_path),
            preview_path=str(preview_path),
            num_instances=len(valid_masks)
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Segmentation failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


def refine_mask(mask: np.ndarray) -> np.ndarray:
    """
    Refine segmentation mask with morphological operations.
    
    Args:
        mask: Binary mask (0-255 uint8)
    
    Returns:
        Refined mask (0-255 uint8)
    """
    # Force to numpy array
    mask = np.array(mask)

    # Remove extra dimensions
    mask = np.squeeze(mask)

    # If still 3D, take first channel
    if mask.ndim == 3:
        mask = mask[:, :, 0]

    if mask.ndim != 2:
        raise ValueError(f"Mask must be 2D, got shape {mask.shape}")

    mask = mask.astype(np.uint8)


    # Morphological closing (fill holes)
    kernel_close = np.ones((7, 7), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close)
    
    # Remove small components
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    
    h, w = mask.shape
    min_area = int(h * w * 0.01)  # 1% of image
    
    mask_filtered = np.zeros_like(mask)
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= min_area:
            mask_filtered[labels == i] = 255
    
    # Smooth edges
    mask_smoothed = cv2.GaussianBlur(mask_filtered, (5, 5), 0)
    _, mask_binary = cv2.threshold(mask_smoothed, 127, 255, cv2.THRESH_BINARY)
    
    # Do NOT apply a second GaussianBlur here.
    # The saved mask must be hard binary (0 or 255) so that:
    #  - canvas destination-in / pixel alpha assignment works correctly
    #  - PIL putalpha() uses clean 0/255 values
    # Soft feathered masks cause semi-transparent player cutouts in the UI.
    return mask_binary


def create_preview(image_path: Path, mask: np.ndarray, output_path: Path):
    """
    Create preview image with mask applied.
    
    Args:
        image_path: Original image path
        mask: Segmentation mask
        output_path: Where to save preview
    """
    # Load original
    image = cv2.imread(str(image_path))
    if image is None:
        return
    
    # Ensure mask matches image dimensions
    if mask.shape[:2] != image.shape[:2]:
        mask = cv2.resize(mask, (image.shape[1], image.shape[0]))
    
    # Create RGBA image
    b, g, r = cv2.split(image)
    rgba = cv2.merge([r, g, b, mask])  # BGR → RGB with alpha
    
    # Save as PNG with transparency
    cv2.imwrite(str(output_path), rgba)


if __name__ == "__main__":
    import uvicorn
    
    print("=" * 60)
    print("Starting SAM3 Segmentation Service")
    print("=" * 60)
    print("Run with: uvicorn sam3_service:app --host 0.0.0.0 --port 8001")
    print("Or: python sam3_service.py")
    print("=" * 60)
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8001,
        log_level="info"
    )