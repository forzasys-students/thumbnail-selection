"""
Flask Client for SAM3 Service
Functions to call SAM3 microservice from Flask (SoccerNet env)

Add these functions to your app.py in SoccerNet311_y26 environment.
"""

import requests
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)

# SAM3 service configuration
SAM3_SERVICE_URL = "http://localhost:8001"
SAM3_TIMEOUT = 30  # seconds


def check_sam3_service_health() -> bool:
    """
    Check if SAM3 service is running and healthy.
    
    Returns:
        bool: True if service is healthy, False otherwise
    """
    try:
        response = requests.get(
            f"{SAM3_SERVICE_URL}/health",
            timeout=5
        )
        return response.status_code == 200
    except requests.RequestException:
        return False


def call_sam3_segmentation(
    image_path: str,
    prompt: str = "soccer player",
    confidence_threshold: float = 0.5,
    output_dir: Optional[str] = None
) -> Dict:
    """
    Call SAM3 microservice to segment players.
    
    Args:
        image_path: Absolute path to image file
        prompt: Text prompt for SAM3 (e.g., "soccer player", "celebrating player")
        confidence_threshold: Minimum confidence score (0-1)
        output_dir: Optional directory for output masks
    
    Returns:
        dict: {
            'success': bool,
            'mask_path': str,
            'preview_path': str,
            'num_instances': int,
            'error': str (optional)
        }
    
    Raises:
        requests.RequestException: If service is unreachable
        ValueError: If service returns error
    """
    # Check service health first
    if not check_sam3_service_health():
        raise ConnectionError(
            "SAM3 service is not running. "
            "Start it with: uvicorn sam3_service:app --host 0.0.0.0 --port 8001"
        )
    
    # Prepare request
    payload = {
        "image_path": image_path,
        "prompt": prompt,
        "confidence_threshold": confidence_threshold
    }
    
    if output_dir:
        payload["output_dir"] = output_dir
    
    logger.info(f"Calling SAM3 service: {image_path}")
    logger.info(f"Prompt: '{prompt}'")
    
    try:
        # Call segmentation endpoint
        response = requests.post(
            f"{SAM3_SERVICE_URL}/segment",
            json=payload,
            timeout=SAM3_TIMEOUT
        )
        
        # Check response
        response.raise_for_status()
        result = response.json()
        
        if result.get('success'):
            logger.info(f"✓ Segmentation successful: {result.get('num_instances')} instance(s)")
            logger.info(f"  Mask: {result.get('mask_path')}")
        else:
            logger.warning(f"Segmentation completed with warning: {result.get('error')}")
        
        return result
        
    except requests.Timeout:
        logger.error("SAM3 service timeout")
        raise TimeoutError(f"SAM3 service did not respond within {SAM3_TIMEOUT}s")
    
    except requests.HTTPError as e:
        logger.error(f"SAM3 service error: {e}")
        error_detail = e.response.json().get('detail', str(e)) if e.response else str(e)
        raise ValueError(f"SAM3 segmentation failed: {error_detail}")
    
    except requests.RequestException as e:
        logger.error(f"Failed to connect to SAM3 service: {e}")
        raise ConnectionError(
            f"Cannot reach SAM3 service at {SAM3_SERVICE_URL}. "
            "Ensure the service is running."
        )


def compose_thumbnail(
    image_path: str,
    mask_path: str,
    elements: list,
    background_color: str = "#000000",
    output_path: str = None,
    # NEW params forwarded from the frontend
    player_layer: str = "foreground",
    blur_background: bool = False,
    blur_radius: int = 12,
) -> str:
    """
    Compose final thumbnail with player cutout and graphic overlays.

    Args:
        image_path:       Original image path
        mask_path:        Segmentation mask path
        elements:         List of graphic element configurations (text, rect, logo)
        background_color: Background color (hex)
        output_path:      Output thumbnail path
        player_layer:     'foreground' (default) or 'background'
        blur_background:  If True, background is blurred but player stays sharp
        blur_radius:      Blur strength in pixels

    Returns:
        str: Path to created thumbnail
    """
    from thumbnail_compositor import ThumbnailCompositor

    compositor = ThumbnailCompositor(debug=False)

    thumbnail_path = compositor.create_thumbnail(
        image_path=image_path,
        mask_path=mask_path,
        elements=elements,
        background_color=background_color,
        output_path=output_path,
        player_layer=player_layer,
        blur_background=blur_background,
        blur_radius=blur_radius,
    )

    return thumbnail_path


# =============================================================================
# Example usage (test script)
# =============================================================================
if __name__ == "__main__":
    import sys
    import logging
    
    logging.basicConfig(level=logging.INFO)
    
    # Test SAM3 service connection
    print("Testing SAM3 service...")
    
    if not check_sam3_service_health():
        print("✗ SAM3 service is not running!")
        print("\nStart the service with:")
        print("  conda activate sam3_env")
        print("  uvicorn sam3_service:app --host 0.0.0.0 --port 8001")
        sys.exit(1)
    
    print("✓ SAM3 service is healthy")
    
    # Test segmentation
    test_image = "data/inference_output/keyframes/rank01_P1_conf0.92_score0.845.jpg"
    
    if len(sys.argv) > 1:
        test_image = sys.argv[1]
    
    print(f"\nTesting segmentation on: {test_image}")
    
    try:
        result = call_sam3_segmentation(
            image_path=test_image,
            prompt="soccer player",
            confidence_threshold=0.5
        )
        
        print("\nResult:")
        print(f"  Success: {result['success']}")
        print(f"  Instances: {result['num_instances']}")
        print(f"  Mask: {result['mask_path']}")
        print(f"  Preview: {result['preview_path']}")
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        sys.exit(1)