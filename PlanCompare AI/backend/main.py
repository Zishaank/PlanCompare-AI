import json
import os
from pathlib import Path
from typing import Any, Dict, List

import cv2
import fitz
import numpy as np
import openai
import pytesseract
from PIL import Image
from pytesseract import Output, TesseractNotFoundError
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware

UPLOAD_DIR = Path(__file__).resolve().parent / "uploads"
PROCESSED_DIR = Path(__file__).resolve().parent / "processed"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="PlanCompare API")

origins = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["POST"],
    allow_headers=["*"],
)

async def save_pdf(file: UploadFile, target_dir: Path) -> Path:
    if file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail=f"File {file.filename} must be a PDF.")

    destination = target_dir / file.filename
    contents = await file.read()
    destination.write_bytes(contents)
    return destination

def convert_first_page_to_png(pdf_path: Path, target_dir: Path) -> Path:
    document = fitz.open(pdf_path)
    if document.page_count < 1:
        raise HTTPException(status_code=400, detail=f"PDF {pdf_path.name} has no pages.")

    page = document.load_page(0)
    zoom = 2.0
    matrix = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=matrix, alpha=False)

    output_path = target_dir / f"{pdf_path.stem}_page1.png"
    pix.save(str(output_path))
    return output_path

def align_new_image_to_old(old_image_path: Path, new_image_path: Path, target_dir: Path) -> Path:
    old_gray = cv2.imread(str(old_image_path), cv2.IMREAD_GRAYSCALE)
    new_gray = cv2.imread(str(new_image_path), cv2.IMREAD_GRAYSCALE)
    if old_gray is None or new_gray is None:
        raise HTTPException(status_code=500, detail="Unable to load one of the images for alignment.")

    orb = cv2.ORB_create(5000)
    keypoints_old, descriptors_old = orb.detectAndCompute(old_gray, None)
    keypoints_new, descriptors_new = orb.detectAndCompute(new_gray, None)

    if descriptors_old is None or descriptors_new is None:
        raise HTTPException(status_code=400, detail="Insufficient features found for alignment.")

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = matcher.match(descriptors_old, descriptors_new)
    if len(matches) < 10:
        raise HTTPException(status_code=400, detail="Not enough matching features to estimate homography.")

    matches = sorted(matches, key=lambda m: m.distance)
    src_pts = np.float32([keypoints_new[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([keypoints_old[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)

    homography, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    if homography is None:
        raise HTTPException(status_code=400, detail="Homography estimation failed.")

    new_color = cv2.imread(str(new_image_path), cv2.IMREAD_COLOR)
    if new_color is None:
        raise HTTPException(status_code=500, detail="Unable to load new image for warping.")

    height, width = old_gray.shape
    aligned = cv2.warpPerspective(new_color, homography, (width, height), flags=cv2.INTER_LINEAR)

    output_path = target_dir / f"{new_image_path.stem}_aligned.png"
    cv2.imwrite(str(output_path), aligned)
    return output_path

def compare_drawing_images(old_image_path: Path, aligned_new_image_path: Path, target_dir: Path):
    old_gray = cv2.imread(str(old_image_path), cv2.IMREAD_GRAYSCALE)
    new_gray = cv2.imread(str(aligned_new_image_path), cv2.IMREAD_GRAYSCALE)
    if old_gray is None or new_gray is None:
        raise HTTPException(status_code=500, detail="Unable to load one of the images for comparison.")

    diff = cv2.absdiff(old_gray, new_gray)
    _, thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)

    kernel = np.ones((5, 5), np.uint8)
    cleaned = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
    cleaned = cv2.dilate(cleaned, kernel, iterations=2)

    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours is None:
        contours = []

    comparison_color = cv2.imread(str(aligned_new_image_path), cv2.IMREAD_COLOR)
    if comparison_color is None:
        raise HTTPException(status_code=500, detail="Unable to load aligned image for markup.")

    boxes = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 300:
            continue

        x, y, w, h = cv2.boundingRect(contour)
        box = {"x": int(x), "y": int(y), "width": int(w), "height": int(h)}
        ocr_result = ocr_text_for_box(aligned_new_image_path, box)
        box["ocr_text"] = ocr_result["text"]
        box["ocr_confidence"] = ocr_result["confidence"]
        boxes.append(box)
        cv2.rectangle(comparison_color, (x, y), (x + w, y + h), (0, 0, 255), 2)

    comparison_path = target_dir / f"{aligned_new_image_path.stem}_comparison.png"
    cv2.imwrite(str(comparison_path), comparison_color)

    return comparison_path, boxes

def ocr_text_for_box(image_path: Path, bbox: dict, padding: int = 20) -> dict:
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(status_code=500, detail="Unable to load image for OCR crop.")

    height, width = image.shape[:2]
    x1 = max(0, bbox["x"] - padding)
    y1 = max(0, bbox["y"] - padding)
    x2 = min(width, bbox["x"] + bbox["width"] + padding)
    y2 = min(height, bbox["y"] + bbox["height"] + padding)

    crop = image[y1:y2, x1:x2]
    if crop.size == 0:
        return {"text": "", "confidence": 0.0}

    crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    pil_crop = Image.fromarray(crop_rgb)
    try:
        ocr_data = pytesseract.image_to_data(pil_crop, output_type=Output.DICT)
    except TesseractNotFoundError:
        return {"text": "Requires manual review", "confidence": 0.0}

    words = []
    confidences = []
    for idx, text in enumerate(ocr_data.get("text", [])):
        stripped = str(text).strip()
        if not stripped:
            continue

        try:
            conf = float(ocr_data.get("conf", [])[idx])
        except (ValueError, IndexError, TypeError):
            continue

        if conf >= 0:
            words.append(stripped)
            confidences.append(conf)

    joined_text = " ".join(words)
    average_confidence = float(np.mean(confidences)) if confidences else 0.0

    return {"text": joined_text, "confidence": round(average_confidence, 2)}

def build_llm_prompt(metadata: Dict[str, Any], bounding_boxes: List[Dict[str, Any]], visual_notes: str) -> str:
    prompt = (
        "You are a construction drawing revision assistant.\n\n"
        "Compare the detected changes between the previous and revised drawing.\n"
        "Use only the provided OCR text and detected change regions.\n"
        "Do not invent changes.\n"
        "If the detected information is unclear, say \"Requires manual review\".\n\n"
        "Generate a concise change report for a site engineer.\n\n"
        "Drawing metadata:\n"
        f"{json.dumps(metadata, indent=2)}\n\n"
        "Detected changes:\n"
    )

    for idx, box in enumerate(bounding_boxes, start=1):
        prompt += (
            f"Change {idx}: bbox={box['x']},{box['y']},{box['width']},{box['height']}. "
            f"OCR text=\"{box.get('ocr_text', '')}\". "
            f"Confidence={box.get('ocr_confidence', 0.0)}.\n"
        )

    prompt += f"\nVisual comparison notes:\n{visual_notes}\n\n"
    prompt += (
        "Respond with JSON containing: executive_summary, changes, construction_impact, "
        "recommended_site_team_checks, assumptions_and_limitations."
    )
    return prompt


def send_changes_to_llm(metadata: Dict[str, Any], bounding_boxes: List[Dict[str, Any]], visual_notes: str) -> Dict[str, Any]:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="OpenAI API key is not configured.")

    openai.api_key = api_key
    prompt = build_llm_prompt(metadata, bounding_boxes, visual_notes)

    response = openai.ChatCompletion.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
        max_tokens=600,
    )

    content = response.choices[0].message.content
    try:
        report = json.loads(content)
    except json.JSONDecodeError:
        raise HTTPException(status_code=500, detail="Failed to parse LLM output as JSON.")

    required_keys = [
        "executive_summary",
        "changes",
        "construction_impact",
        "recommended_site_team_checks",
        "assumptions_and_limitations",
    ]
    if not all(key in report for key in required_keys):
        raise HTTPException(status_code=500, detail="LLM response is missing required report sections.")

    return report

@app.post("/compare-drawings")
async def compare_drawings(
    old_file: UploadFile = File(...),
    new_file: UploadFile = File(...),
):
    old_path = await save_pdf(old_file, UPLOAD_DIR)
    new_path = await save_pdf(new_file, UPLOAD_DIR)

    old_image = convert_first_page_to_png(old_path, PROCESSED_DIR)
    new_image = convert_first_page_to_png(new_path, PROCESSED_DIR)
    aligned_new_image = align_new_image_to_old(old_image, new_image, PROCESSED_DIR)
    comparison_image, bounding_boxes = compare_drawing_images(old_image, aligned_new_image, PROCESSED_DIR)

    drawing_metadata = {
        "old_file": old_path.name,
        "new_file": new_path.name,
        "old_image": old_image.name,
        "new_image": new_image.name,
        "aligned_new_image": aligned_new_image.name,
        "comparison_image": comparison_image.name,
        "upload_directory": str(UPLOAD_DIR),
        "processed_directory": str(PROCESSED_DIR),
    }

    visual_notes = (
        f"Detected {len(bounding_boxes)} changed regions in the aligned comparison image. "
        "OCR results are attached to each bounding box."
    )

    llm_report = send_changes_to_llm(drawing_metadata, bounding_boxes, visual_notes)

    return {
        "message": "Both PDF files were received, saved, processed, aligned, compared, and summarized successfully.",
        "files": {
            "old_file": old_path.name,
            "new_file": new_path.name,
            "old_image": old_image.name,
            "new_image": new_image.name,
            "aligned_new_image": aligned_new_image.name,
            "comparison_image": comparison_image.name,
        },
        "bounding_boxes": bounding_boxes,
        "llm_report": llm_report,
        "upload_directory": str(UPLOAD_DIR),
        "processed_directory": str(PROCESSED_DIR),
    }
