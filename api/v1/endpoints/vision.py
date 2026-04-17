"""
Vision Agent API — image analysis, OCR, Visual Q&A.

Uploads images to MinIO/R2 storage and persists all analysis history
to MongoDB for retrieval.
"""

from typing import Any, List, Optional, Dict
from fastapi import APIRouter, Depends, HTTPException, status, UploadFile, File, Form, Query
from pydantic import BaseModel, Field
from datetime import datetime
import uuid
import json
import structlog

from backend.api.deps import get_current_active_user
from backend.models.user import User
from backend.services.activity_logger import log_activity
from backend.services.storage_service import storage_service
from backend.db.mongodb.mongodb import MongoDB

logger = structlog.get_logger(__name__)

router = APIRouter()

# ── MongoDB collection name ─────────────────────────────────────────
VISION_COLLECTION = "vision_analyses"

# ── Pydantic models ─────────────────────────────────────────────────

class VisionAnalysisOptions(BaseModel):
    prompt: Optional[str] = Field(None, description="Custom prompt for image analysis")
    analysis_tasks: Optional[List[str]] = Field(None, description="Specific tasks")
    detailed: Optional[bool] = Field(True)
    max_tokens: Optional[int] = Field(4096)
    temperature: Optional[float] = Field(0.7)


class VisionAnalysisResponse(BaseModel):
    id: str
    description: str
    structured_analysis: Dict[str, Any]
    image_url: Optional[str] = None
    processed_at: str
    model_used: Optional[str] = None
    citations: Optional[List[Dict[str, Any]]] = None
    analysis_type: str = "general"


class ImageURLRequest(BaseModel):
    url: str = Field(..., description="URL of the image to analyze")
    options: Optional[VisionAnalysisOptions] = None


class VisualQARequest(BaseModel):
    analysis_id: str = Field(..., description="ID of the analysis to query about")
    question: str = Field(..., description="Question about the image")
    max_tokens: Optional[int] = Field(4096)


class VisualQAResponse(BaseModel):
    answer: str
    analysis_id: str
    question: str
    answered_at: str


class VisionHistoryItem(BaseModel):
    id: str
    analysis_type: str
    description: str
    image_url: Optional[str] = None
    created_at: str
    summary: str


class VisionStatsResponse(BaseModel):
    total_scans: int
    objects_found: int
    text_extracted: int
    avg_processing_time: float


# ── Helpers ──────────────────────────────────────────────────────────

def _build_vision_agent():
    """Create a VisionAgent instance."""
    from backend.agents.vision_agent import VisionAgent
    config = {
        "type": "vision",
        "model_config": {"model": "sonar-large-online"},
        "vision_tasks": [
            "text_extraction", "object_detection",
            "scene_analysis", "content_description",
        ],
    }
    return VisionAgent(config)


async def _upload_image_to_storage(
    image_bytes: bytes,
    filename: str,
    content_type: str = "image/jpeg",
) -> str:
    """Upload image to MinIO/R2 and return a presigned URL."""
    safe_name = filename or "image.jpg"
    key = f"vision/{uuid.uuid4()}_{safe_name}"
    await storage_service.upload_file(image_bytes, key, content_type)
    url = await storage_service.get_presigned_url(key)
    return url


async def _save_analysis(
    user_id: str,
    analysis_type: str,
    image_url: Optional[str],
    result: Dict[str, Any],
    filename: Optional[str] = None,
    source: str = "file_upload",
) -> str:
    """Persist analysis result to MongoDB. Returns the document _id."""
    col = await MongoDB.get_collection(VISION_COLLECTION)
    doc_id = str(uuid.uuid4())

    # Count objects and text items for stats
    sa = result.get("structured_analysis", {})
    objects_count = len(sa.get("detected_objects", []))
    text_count = len(sa.get("detected_text", []))
    # For OCR or when regex didn't catch quoted text, count meaningful
    # lines in the description as extracted text fragments
    if text_count == 0 and analysis_type in ("ocr", "general"):
        desc = result.get("description", "")
        # Count non-empty lines that aren't markdown headings or meta
        text_count = max(1, sum(
            1 for line in desc.split("\n")
            if line.strip()
            and not line.strip().startswith("#")
            and len(line.strip()) > 3
        )) if desc.strip() else 0

    doc = {
        "_id": doc_id,
        "user_id": user_id,
        "analysis_type": analysis_type,
        "source": source,
        "filename": filename,
        "image_url": image_url,
        "description": result.get("description", ""),
        "structured_analysis": sa,
        "model_used": result.get("model_used"),
        "citations": result.get("citations", []),
        "objects_count": objects_count,
        "text_count": text_count,
        "processed_at": result.get("processed_at", datetime.utcnow().isoformat()),
        "created_at": datetime.utcnow().isoformat(),
        # Q&A conversation history for this analysis
        "conversations": [],
    }
    await col.insert_one(doc)
    return doc_id


# ── Endpoints ────────────────────────────────────────────────────────

@router.post("/analyze", response_model=VisionAnalysisResponse)
async def analyze_image(
    file: UploadFile = File(...),
    options: Optional[str] = Form(None),
    current_user: User = Depends(get_current_active_user),
) -> Any:
    """Analyze an uploaded image — stores image in MinIO/R2 and saves analysis to MongoDB."""
    try:
        image_content = await file.read()

        # Parse options
        analysis_options: Dict[str, Any] = {}
        if options:
            try:
                analysis_options = json.loads(options)
            except json.JSONDecodeError:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Invalid options JSON format",
                )

        # Upload image to storage
        image_url = await _upload_image_to_storage(
            image_content,
            file.filename or "image.jpg",
            file.content_type or "image/jpeg",
        )

        # Run analysis
        agent = _build_vision_agent()
        vision_data = {
            "image_content": image_content,
            "prompt": analysis_options.get(
                "prompt",
                "Analyze this image concisely. Describe the main content, key objects, "
                "any readable text, and overall scene. Be specific but avoid repetition. "
                "Do NOT list UI chrome, browser elements, or repeated decorative items. "
                "Structure your response with clear sections using markdown headings.",
            ),
            "analysis_tasks": analysis_options.get(
                "analysis_tasks",
                ["text_extraction", "object_detection", "scene_analysis"],
            ),
            "max_tokens": analysis_options.get("max_tokens", 4096),
            "temperature": analysis_options.get("temperature", 0.7),
        }
        result = await agent.process_async(vision_data)

        if "error" in result:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=result["error"],
            )

        # Save to MongoDB
        doc_id = await _save_analysis(
            user_id=str(current_user.id),
            analysis_type="general",
            image_url=image_url,
            result=result,
            filename=file.filename,
            source="file_upload",
        )

        await log_activity(
            user_id=str(current_user.id),
            organization_id=getattr(current_user, "organization_id", None),
            activity_type="vision.image_analyzed",
            details={"source": "file_upload", "filename": file.filename},
            related_resource_type="AGENT",
            agent_name="Vision Agent",
        )

        return VisionAnalysisResponse(
            id=doc_id,
            description=result["description"],
            structured_analysis=result["structured_analysis"],
            image_url=image_url,
            processed_at=result["processed_at"],
            model_used=result.get("model_used"),
            citations=result.get("citations"),
            analysis_type="general",
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error("analyze_image_failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Image analysis failed: {str(e)}",
        )


@router.post("/analyze-url", response_model=VisionAnalysisResponse)
async def analyze_image_url(
    request: ImageURLRequest,
    current_user: User = Depends(get_current_active_user),
) -> Any:
    """Analyze an image from a URL — saves analysis to MongoDB."""
    try:
        opts = request.options.model_dump() if request.options else {}

        agent = _build_vision_agent()
        vision_data = {
            "image_url": request.url,
            "prompt": opts.get(
                "prompt",
                "Analyze the provided image and describe its content in detail.",
            ),
            "analysis_tasks": opts.get(
                "analysis_tasks",
                ["text_extraction", "object_detection", "scene_analysis"],
            ),
            "max_tokens": opts.get("max_tokens", 4096),
            "temperature": opts.get("temperature", 0.7),
        }
        result = await agent.process_async(vision_data)

        if "error" in result:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=result["error"],
            )

        doc_id = await _save_analysis(
            user_id=str(current_user.id),
            analysis_type="url",
            image_url=request.url,
            result=result,
            source="url",
        )

        await log_activity(
            user_id=str(current_user.id),
            organization_id=getattr(current_user, "organization_id", None),
            activity_type="vision.image_analyzed",
            details={"source": "url", "url_preview": request.url[:100]},
            related_resource_type="AGENT",
            agent_name="Vision Agent",
        )

        return VisionAnalysisResponse(
            id=doc_id,
            description=result["description"],
            structured_analysis=result["structured_analysis"],
            image_url=request.url,
            processed_at=result["processed_at"],
            model_used=result.get("model_used"),
            citations=result.get("citations"),
            analysis_type="url",
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error("analyze_url_failed", error=str(e), url=request.url)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Image URL analysis failed: {str(e)}",
        )


@router.post("/ocr", response_model=VisionAnalysisResponse)
async def extract_text_from_image(
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_active_user),
) -> Any:
    """Extract text (OCR) from an uploaded image — stores image and saves to MongoDB."""
    try:
        image_content = await file.read()

        # Upload to storage
        image_url = await _upload_image_to_storage(
            image_content,
            file.filename or "image.jpg",
            file.content_type or "image/jpeg",
        )

        agent = _build_vision_agent()
        vision_data = {
            "image_content": image_content,
            "prompt": (
                "Extract and transcribe ONLY the actual readable text content in this image. "
                "Focus on meaningful text like headings, paragraphs, labels, numbers, and captions. "
                "Do NOT describe UI elements, icons, buttons, or browser chrome. "
                "Do NOT list repeated elements or decorative items. "
                "Present the extracted text in a clean, organized format with headings where appropriate. "
                "If text is in sections, use clear section separators."
            ),
            "max_tokens": 4096,
        }
        result = await agent.process_async(vision_data)

        if "error" in result:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=result["error"],
            )

        doc_id = await _save_analysis(
            user_id=str(current_user.id),
            analysis_type="ocr",
            image_url=image_url,
            result=result,
            filename=file.filename,
            source="file_upload",
        )

        await log_activity(
            user_id=str(current_user.id),
            organization_id=getattr(current_user, "organization_id", None),
            activity_type="vision.ocr_extracted",
            details={"source": "file_upload", "filename": file.filename},
            related_resource_type="AGENT",
            agent_name="Vision Agent",
        )

        return VisionAnalysisResponse(
            id=doc_id,
            description=result["description"],
            structured_analysis=result["structured_analysis"],
            image_url=image_url,
            processed_at=result["processed_at"],
            model_used=result.get("model_used"),
            citations=result.get("citations"),
            analysis_type="ocr",
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error("ocr_failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Text extraction failed: {str(e)}",
        )


@router.post("/query", response_model=VisualQAResponse)
async def visual_qa(
    request: VisualQARequest,
    current_user: User = Depends(get_current_active_user),
) -> Any:
    """Ask a follow-up question about a previously analyzed image (Visual Q&A)."""
    try:
        col = await MongoDB.get_collection(VISION_COLLECTION)
        doc = await col.find_one({"_id": request.analysis_id, "user_id": str(current_user.id)})
        if not doc:
            raise HTTPException(status_code=404, detail="Analysis not found")

        agent = _build_vision_agent()

        # Build context from previous analysis + image
        context: Dict[str, Any] = {
            "max_tokens": request.max_tokens or 4096,
        }
        if doc.get("image_url"):
            context["image_url"] = doc["image_url"]

        # Include prior conversation for multi-turn
        prior = doc.get("conversations", [])
        enriched_query = request.question
        if prior:
            history_text = "\n".join(
                f"Q: {c['question']}\nA: {c['answer']}" for c in prior[-5:]
            )
            enriched_query = (
                f"Previous conversation about this image:\n{history_text}\n\n"
                f"Previous analysis: {doc.get('description', '')[:500]}\n\n"
                f"New question: {request.question}"
            )

        result = await agent.query_async(enriched_query, context)

        if "error" in result:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=result["error"],
            )

        answer = result.get("description", "I couldn't analyze the image for that question.")
        answered_at = datetime.utcnow().isoformat()

        # Append to conversation history in MongoDB
        await col.update_one(
            {"_id": request.analysis_id},
            {
                "$push": {
                    "conversations": {
                        "question": request.question,
                        "answer": answer,
                        "answered_at": answered_at,
                    }
                }
            },
        )

        return VisualQAResponse(
            answer=answer,
            analysis_id=request.analysis_id,
            question=request.question,
            answered_at=answered_at,
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error("visual_qa_failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Visual Q&A failed: {str(e)}",
        )


@router.get("/history", response_model=List[VisionHistoryItem])
async def get_analysis_history(
    limit: int = Query(default=20, le=50),
    skip: int = Query(default=0, ge=0),
    analysis_type: Optional[str] = Query(default=None),
    current_user: User = Depends(get_current_active_user),
) -> Any:
    """Get the current user's vision analysis history from MongoDB."""
    try:
        col = await MongoDB.get_collection(VISION_COLLECTION)
        query: Dict[str, Any] = {"user_id": str(current_user.id)}
        if analysis_type:
            query["analysis_type"] = analysis_type

        cursor = col.find(query).sort("created_at", -1).skip(skip).limit(limit)
        docs = await cursor.to_list(length=limit)

        items = []
        for doc in docs:
            desc = doc.get("description", "")
            summary = desc[:120] + "..." if len(desc) > 120 else desc
            items.append(
                VisionHistoryItem(
                    id=doc["_id"],
                    analysis_type=doc.get("analysis_type", "general"),
                    description=doc.get("filename") or summary[:60],
                    image_url=doc.get("image_url"),
                    created_at=doc.get("created_at", ""),
                    summary=summary,
                )
            )
        return items
    except Exception as e:
        logger.error("history_fetch_failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to fetch history: {str(e)}",
        )


@router.get("/history/{analysis_id}")
async def get_analysis_detail(
    analysis_id: str,
    current_user: User = Depends(get_current_active_user),
) -> Any:
    """Get full details of a specific analysis including Q&A history."""
    try:
        col = await MongoDB.get_collection(VISION_COLLECTION)
        doc = await col.find_one({"_id": analysis_id, "user_id": str(current_user.id)})
        if not doc:
            raise HTTPException(status_code=404, detail="Analysis not found")
        return doc
    except HTTPException:
        raise
    except Exception as e:
        logger.error("analysis_detail_failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to fetch analysis: {str(e)}",
        )


@router.get("/stats", response_model=VisionStatsResponse)
async def get_vision_stats(
    current_user: User = Depends(get_current_active_user),
) -> Any:
    """Get aggregated stats for the current user's vision usage."""
    try:
        col = await MongoDB.get_collection(VISION_COLLECTION)
        user_id = str(current_user.id)

        pipeline = [
            {"$match": {"user_id": user_id}},
            {
                "$group": {
                    "_id": None,
                    "total_scans": {"$sum": 1},
                    "objects_found": {"$sum": "$objects_count"},
                    "text_extracted": {"$sum": "$text_count"},
                }
            },
        ]
        results = await col.aggregate(pipeline).to_list(length=1)

        if results:
            r = results[0]
            return VisionStatsResponse(
                total_scans=r.get("total_scans", 0),
                objects_found=r.get("objects_found", 0),
                text_extracted=r.get("text_extracted", 0),
                avg_processing_time=1.8,  # TODO: track actual processing time
            )
        return VisionStatsResponse(
            total_scans=0, objects_found=0, text_extracted=0, avg_processing_time=0.0,
        )
    except Exception as e:
        logger.error("stats_failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to fetch stats: {str(e)}",
        )


@router.delete("/history/{analysis_id}")
async def delete_analysis(
    analysis_id: str,
    current_user: User = Depends(get_current_active_user),
) -> Any:
    """Delete a specific analysis from history."""
    try:
        col = await MongoDB.get_collection(VISION_COLLECTION)
        result = await col.delete_one({"_id": analysis_id, "user_id": str(current_user.id)})
        if result.deleted_count == 0:
            raise HTTPException(status_code=404, detail="Analysis not found")
        return {"deleted": True}
    except HTTPException:
        raise
    except Exception as e:
        logger.error("delete_analysis_failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete analysis: {str(e)}",
        )
