"""
Blog CRUD endpoints — public listing + authenticated create/update/delete.
Posts are stored in Postgres with denormalized author metadata.
"""

import re
import random
import string
from datetime import datetime
from typing import List, Optional

import bleach
from bleach.css_sanitizer import CSSSanitizer
from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, Field
from sqlalchemy import select, func, update, desc
from sqlalchemy.ext.asyncio import AsyncSession
import structlog

from backend.api.deps import get_current_active_user, get_current_superuser
from backend.models.user import User
from backend.db.postgres import get_async_sessionmaker
from backend.db.postgres_models import BlogPostSQL, BlogPostStatus, AuthorType, BlogCommentSQL

logger = structlog.get_logger(__name__)

router = APIRouter()

# ── HTML Sanitization ─────────────────────────────────────────────

ALLOWED_TAGS = [
    "p", "br", "strong", "em", "u", "s", "a", "img", "h1", "h2", "h3", "h4",
    "ul", "ol", "li", "blockquote", "pre", "code", "hr", "div", "span",
    "figure", "figcaption", "iframe", "table", "thead", "tbody", "tr", "th", "td",
]
ALLOWED_ATTRS = {
    "*": ["class", "id", "style"],
    "a": ["href", "target", "rel"],
    "img": ["src", "alt", "width", "height"],
    "iframe": ["src", "width", "height", "frameborder", "allowfullscreen", "allow"],
}


css_sanitizer = CSSSanitizer(allowed_css_properties=["color", "font-weight", "text-align", "margin", "padding"])


def sanitize_html(html: str) -> str:
    return bleach.clean(
        html,
        tags=ALLOWED_TAGS,
        attributes=ALLOWED_ATTRS,
        css_sanitizer=css_sanitizer,
        strip=True,
    )


# ── Slug Generation ───────────────────────────────────────────────

def slugify(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^\w\s-]", "", text)
    text = re.sub(r"[\s_]+", "-", text)
    text = re.sub(r"-+", "-", text).strip("-")
    return text[:200]


def generate_slug(title: str) -> str:
    base = slugify(title)
    date_suffix = datetime.utcnow().strftime("%Y%m%d")
    return f"{base}-{date_suffix}"


def random_suffix(length: int = 4) -> str:
    return "".join(random.choices(string.ascii_lowercase + string.digits, k=length))


# ── Request / Response Models ─────────────────────────────────────

class BlogPostCreate(BaseModel):
    title: str = Field(..., max_length=500)
    subtitle: Optional[str] = Field(None, max_length=500)
    content: str
    excerpt: Optional[str] = None
    cover_image_url: Optional[str] = None
    category: Optional[str] = None
    tags: List[str] = Field(default_factory=list)
    status: BlogPostStatus = BlogPostStatus.DRAFT
    author_type: Optional[AuthorType] = None
    featured: bool = False


class BlogPostUpdate(BaseModel):
    title: Optional[str] = Field(None, max_length=500)
    subtitle: Optional[str] = Field(None, max_length=500)
    content: Optional[str] = None
    excerpt: Optional[str] = None
    cover_image_url: Optional[str] = None
    category: Optional[str] = None
    tags: Optional[List[str]] = None
    status: Optional[BlogPostStatus] = None
    featured: Optional[bool] = None


class CollaboratorAdd(BaseModel):
    user_id: str


class BlogPostResponse(BaseModel):
    id: str
    slug: str
    title: str
    subtitle: Optional[str] = None
    content: str
    excerpt: Optional[str] = None
    author_id: str
    author_type: str
    author_name: str
    author_avatar_url: Optional[str] = None
    author_title: Optional[str] = None
    cover_image_url: Optional[str] = None
    category: Optional[str] = None
    tags: List[str] = []
    status: str
    collaborator_ids: List[str] = []
    featured: bool = False
    view_count: int = 0
    published_at: Optional[str] = None
    created_at: str
    updated_at: str

    model_config = {"from_attributes": True}


class BlogPostListResponse(BaseModel):
    posts: List[BlogPostResponse]
    total: int
    page: int
    page_size: int


class CategoryCount(BaseModel):
    category: str
    count: int


def _post_to_response(post: BlogPostSQL) -> BlogPostResponse:
    return BlogPostResponse(
        id=post.id,
        slug=post.slug,
        title=post.title,
        subtitle=post.subtitle,
        content=post.content,
        excerpt=post.excerpt,
        author_id=post.author_id,
        author_type=post.author_type.value if post.author_type else "individual",
        author_name=post.author_name,
        author_avatar_url=post.author_avatar_url,
        author_title=post.author_title,
        cover_image_url=post.cover_image_url,
        category=post.category,
        tags=post.tags or [],
        status=post.status.value if post.status else "draft",
        collaborator_ids=post.collaborator_ids or [],
        featured=post.featured or False,
        view_count=post.view_count or 0,
        published_at=post.published_at.isoformat() if post.published_at else None,
        created_at=post.created_at.isoformat() if post.created_at else "",
        updated_at=post.updated_at.isoformat() if post.updated_at else "",
    )


# ── Public Endpoints ──────────────────────────────────────────────

@router.get("", response_model=BlogPostListResponse)
async def list_posts(
    page: int = Query(1, ge=1),
    page_size: int = Query(12, ge=1, le=50),
    category: Optional[str] = None,
    tag: Optional[str] = None,
    search: Optional[str] = None,
    featured: Optional[bool] = None,
):
    """Public blog listing — paginated, filterable."""
    sf = get_async_sessionmaker()
    async with sf() as session:
        query = select(BlogPostSQL).where(
            BlogPostSQL.status == BlogPostStatus.PUBLISHED,
            BlogPostSQL.deleted_at.is_(None),
        )
        count_query = select(func.count(BlogPostSQL.id)).where(
            BlogPostSQL.status == BlogPostStatus.PUBLISHED,
            BlogPostSQL.deleted_at.is_(None),
        )

        if category:
            query = query.where(BlogPostSQL.category == category)
            count_query = count_query.where(BlogPostSQL.category == category)
        if tag:
            query = query.where(BlogPostSQL.tags.any(tag))
            count_query = count_query.where(BlogPostSQL.tags.any(tag))
        if search:
            like_term = f"%{search}%"
            query = query.where(
                BlogPostSQL.title.ilike(like_term) | BlogPostSQL.excerpt.ilike(like_term)
            )
            count_query = count_query.where(
                BlogPostSQL.title.ilike(like_term) | BlogPostSQL.excerpt.ilike(like_term)
            )
        if featured is not None:
            query = query.where(BlogPostSQL.featured == featured)
            count_query = count_query.where(BlogPostSQL.featured == featured)

        total = (await session.execute(count_query)).scalar() or 0

        query = query.order_by(desc(BlogPostSQL.published_at))
        query = query.offset((page - 1) * page_size).limit(page_size)

        result = await session.execute(query)
        posts = result.scalars().all()

        return BlogPostListResponse(
            posts=[_post_to_response(p) for p in posts],
            total=total,
            page=page,
            page_size=page_size,
        )


@router.get("/categories", response_model=List[CategoryCount])
async def list_categories():
    """Distinct categories with post counts."""
    sf = get_async_sessionmaker()
    async with sf() as session:
        query = (
            select(BlogPostSQL.category, func.count(BlogPostSQL.id).label("count"))
            .where(
                BlogPostSQL.status == BlogPostStatus.PUBLISHED,
                BlogPostSQL.deleted_at.is_(None),
                BlogPostSQL.category.isnot(None),
            )
            .group_by(BlogPostSQL.category)
            .order_by(desc("count"))
        )
        result = await session.execute(query)
        rows = result.all()
        return [CategoryCount(category=r[0], count=r[1]) for r in rows]


@router.get("/my-posts", response_model=BlogPostListResponse)
async def my_posts(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=50),
    current_user: User = Depends(get_current_active_user),
):
    """Current user's posts (all statuses)."""
    user_id = str(current_user.id)
    sf = get_async_sessionmaker()
    async with sf() as session:
        base_where = [
            BlogPostSQL.deleted_at.is_(None),
            (BlogPostSQL.author_id == user_id) | BlogPostSQL.collaborator_ids.any(user_id),
        ]
        total = (
            await session.execute(
                select(func.count(BlogPostSQL.id)).where(*base_where)
            )
        ).scalar() or 0

        result = await session.execute(
            select(BlogPostSQL)
            .where(*base_where)
            .order_by(desc(BlogPostSQL.updated_at))
            .offset((page - 1) * page_size)
            .limit(page_size)
        )
        posts = result.scalars().all()

        return BlogPostListResponse(
            posts=[_post_to_response(p) for p in posts],
            total=total,
            page=page,
            page_size=page_size,
        )


@router.get("/{slug}", response_model=BlogPostResponse)
async def get_post(slug: str):
    """Get a single post by slug. Increments view count."""
    sf = get_async_sessionmaker()
    async with sf() as session:
        result = await session.execute(
            select(BlogPostSQL).where(
                BlogPostSQL.slug == slug,
                BlogPostSQL.deleted_at.is_(None),
            )
        )
        post = result.scalar_one_or_none()
        if not post:
            raise HTTPException(status_code=404, detail="Post not found")

        # Only count views for published posts
        if post.status == BlogPostStatus.PUBLISHED:
            post.view_count = (post.view_count or 0) + 1
            await session.commit()

        return _post_to_response(post)


# ── Authenticated Endpoints ───────────────────────────────────────

@router.post("", response_model=BlogPostResponse, status_code=201)
async def create_post(
    body: BlogPostCreate,
    current_user: User = Depends(get_current_active_user),
):
    """Create a new blog post."""
    sf = get_async_sessionmaker()
    async with sf() as session:
        # Generate unique slug
        slug = generate_slug(body.title)
        existing = await session.execute(
            select(BlogPostSQL.id).where(BlogPostSQL.slug == slug)
        )
        if existing.scalar_one_or_none():
            slug = f"{slug}-{random_suffix()}"

        # Author type: only superusers can set team/ai_agent
        author_type = AuthorType.INDIVIDUAL
        if body.author_type and current_user.is_superuser:
            author_type = body.author_type

        now = datetime.utcnow()
        post = BlogPostSQL(
            slug=slug,
            title=body.title,
            subtitle=body.subtitle,
            content=sanitize_html(body.content),
            excerpt=body.excerpt,
            author_id=str(current_user.id),
            author_type=author_type,
            author_name=current_user.full_name or "Anonymous",
            author_avatar_url=getattr(current_user, "avatar_url", None),
            author_title=getattr(current_user, "job_title", None),
            cover_image_url=body.cover_image_url,
            category=body.category,
            tags=body.tags,
            status=body.status,
            featured=body.featured if current_user.is_superuser else False,
            published_at=now if body.status == BlogPostStatus.PUBLISHED else None,
            created_at=now,
            updated_at=now,
        )
        session.add(post)
        await session.commit()
        await session.refresh(post)

        # Trigger notifications for news/release posts
        if post.status == BlogPostStatus.PUBLISHED and post.category in ("news", "release"):
            await _notify_blog_published(post)

        return _post_to_response(post)


@router.put("/{post_id}", response_model=BlogPostResponse)
async def update_post(
    post_id: str,
    body: BlogPostUpdate,
    current_user: User = Depends(get_current_active_user),
):
    """Update a blog post. Only author, collaborators, or superusers."""
    user_id = str(current_user.id)
    sf = get_async_sessionmaker()
    async with sf() as session:
        result = await session.execute(
            select(BlogPostSQL).where(
                BlogPostSQL.id == post_id,
                BlogPostSQL.deleted_at.is_(None),
            )
        )
        post = result.scalar_one_or_none()
        if not post:
            raise HTTPException(status_code=404, detail="Post not found")

        if (
            post.author_id != user_id
            and user_id not in (post.collaborator_ids or [])
            and not current_user.is_superuser
        ):
            raise HTTPException(status_code=403, detail="Not authorized to edit this post")

        was_draft = post.status != BlogPostStatus.PUBLISHED

        update_data = body.model_dump(exclude_unset=True)
        if "content" in update_data:
            update_data["content"] = sanitize_html(update_data["content"])

        for field, value in update_data.items():
            setattr(post, field, value)

        # Set published_at when first published
        if was_draft and post.status == BlogPostStatus.PUBLISHED and not post.published_at:
            post.published_at = datetime.utcnow()

        post.updated_at = datetime.utcnow()
        await session.commit()
        await session.refresh(post)

        # Notify on first publish
        if was_draft and post.status == BlogPostStatus.PUBLISHED and post.category in ("news", "release"):
            await _notify_blog_published(post)

        return _post_to_response(post)


@router.delete("/{post_id}", status_code=204)
async def delete_post(
    post_id: str,
    current_user: User = Depends(get_current_active_user),
):
    """Soft-delete a blog post. Author or superuser only."""
    user_id = str(current_user.id)
    sf = get_async_sessionmaker()
    async with sf() as session:
        result = await session.execute(
            select(BlogPostSQL).where(
                BlogPostSQL.id == post_id,
                BlogPostSQL.deleted_at.is_(None),
            )
        )
        post = result.scalar_one_or_none()
        if not post:
            raise HTTPException(status_code=404, detail="Post not found")
        if post.author_id != user_id and not current_user.is_superuser:
            raise HTTPException(status_code=403, detail="Not authorized to delete this post")

        post.deleted_at = datetime.utcnow()
        await session.commit()


@router.post("/{post_id}/collaborators", response_model=BlogPostResponse)
async def add_collaborator(
    post_id: str,
    body: CollaboratorAdd,
    current_user: User = Depends(get_current_active_user),
):
    """Add a collaborator to a post. Author only."""
    user_id = str(current_user.id)
    sf = get_async_sessionmaker()
    async with sf() as session:
        result = await session.execute(
            select(BlogPostSQL).where(
                BlogPostSQL.id == post_id,
                BlogPostSQL.deleted_at.is_(None),
            )
        )
        post = result.scalar_one_or_none()
        if not post:
            raise HTTPException(status_code=404, detail="Post not found")
        if post.author_id != user_id and not current_user.is_superuser:
            raise HTTPException(status_code=403, detail="Only the author can add collaborators")

        collabs = list(post.collaborator_ids or [])
        if body.user_id not in collabs:
            collabs.append(body.user_id)
            post.collaborator_ids = collabs
            post.updated_at = datetime.utcnow()
            await session.commit()
            await session.refresh(post)

        return _post_to_response(post)


# ── AI Generate Endpoint ──────────────────────────────────────────

class AIGenerateRequest(BaseModel):
    topic: str = Field(..., max_length=1000)
    category: str = Field(default="engineering", max_length=100)
    auto_publish: bool = False
    cover_image_url: Optional[str] = None


@router.post("/ai-generate", response_model=BlogPostResponse, status_code=201)
async def ai_generate_post(
    body: AIGenerateRequest,
    current_user: User = Depends(get_current_superuser),
):
    """Generate a blog post using AI. Superuser only."""
    from backend.services.blog_writer_service import blog_writer_service

    try:
        post = await blog_writer_service.generate_and_publish(
            topic=body.topic,
            category=body.category,
            auto_publish=body.auto_publish,
            cover_image_url=body.cover_image_url,
        )
        return _post_to_response(post)
    except Exception as e:
        logger.error("ai_blog_generation_failed", error=str(e), topic=body.topic)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"AI generation failed: {str(e)}",
        )


# ── Comment Models ────────────────────────────────────────────────

class MentionItem(BaseModel):
    type: str  # "user" or "agent"
    id: str
    name: str


class CommentCreate(BaseModel):
    content: str = Field(..., max_length=5000)
    mentions: List[MentionItem] = Field(default_factory=list)
    parent_id: Optional[str] = None


class CommentResponse(BaseModel):
    id: str
    post_id: str
    user_id: str
    user_name: str
    user_avatar_url: Optional[str] = None
    content: str
    mentions: list = []
    parent_id: Optional[str] = None
    created_at: str
    updated_at: str

    model_config = {"from_attributes": True}


class CommentListResponse(BaseModel):
    comments: List[CommentResponse]
    total: int


def _comment_to_response(c: BlogCommentSQL) -> CommentResponse:
    return CommentResponse(
        id=c.id,
        post_id=c.post_id,
        user_id=c.user_id,
        user_name=c.user_name,
        user_avatar_url=c.user_avatar_url,
        content=c.content,
        mentions=c.mentions or [],
        parent_id=c.parent_id,
        created_at=c.created_at.isoformat() if c.created_at else "",
        updated_at=c.updated_at.isoformat() if c.updated_at else "",
    )


# ── Comment Endpoints ─────────────────────────────────────────────

@router.get("/{post_id}/comments", response_model=CommentListResponse)
async def list_comments(post_id: str):
    """Get all comments for a post (public)."""
    sf = get_async_sessionmaker()
    async with sf() as session:
        count_q = select(func.count(BlogCommentSQL.id)).where(
            BlogCommentSQL.post_id == post_id,
            BlogCommentSQL.deleted_at.is_(None),
        )
        total = (await session.execute(count_q)).scalar() or 0

        result = await session.execute(
            select(BlogCommentSQL)
            .where(
                BlogCommentSQL.post_id == post_id,
                BlogCommentSQL.deleted_at.is_(None),
            )
            .order_by(BlogCommentSQL.created_at)
        )
        comments = result.scalars().all()
        return CommentListResponse(
            comments=[_comment_to_response(c) for c in comments],
            total=total,
        )


@router.post("/{post_id}/comments", response_model=CommentResponse, status_code=201)
async def create_comment(
    post_id: str,
    body: CommentCreate,
    current_user: User = Depends(get_current_active_user),
):
    """Add a comment to a post. Must be logged in."""
    sf = get_async_sessionmaker()
    async with sf() as session:
        # Verify post exists
        post = await session.execute(
            select(BlogPostSQL.id).where(
                BlogPostSQL.id == post_id,
                BlogPostSQL.deleted_at.is_(None),
            )
        )
        if not post.scalar_one_or_none():
            raise HTTPException(status_code=404, detail="Post not found")

        now = datetime.utcnow()
        comment = BlogCommentSQL(
            post_id=post_id,
            user_id=str(current_user.id),
            user_name=current_user.full_name or "Anonymous",
            user_avatar_url=getattr(current_user, "avatar_url", None),
            content=body.content,
            mentions=[m.model_dump() for m in body.mentions],
            parent_id=body.parent_id,
            created_at=now,
            updated_at=now,
        )
        session.add(comment)
        await session.commit()
        await session.refresh(comment)
        return _comment_to_response(comment)


@router.delete("/{post_id}/comments/{comment_id}", status_code=204)
async def delete_comment(
    post_id: str,
    comment_id: str,
    current_user: User = Depends(get_current_active_user),
):
    """Delete own comment or superuser can delete any."""
    user_id = str(current_user.id)
    sf = get_async_sessionmaker()
    async with sf() as session:
        result = await session.execute(
            select(BlogCommentSQL).where(
                BlogCommentSQL.id == comment_id,
                BlogCommentSQL.post_id == post_id,
                BlogCommentSQL.deleted_at.is_(None),
            )
        )
        comment = result.scalar_one_or_none()
        if not comment:
            raise HTTPException(status_code=404, detail="Comment not found")
        if comment.user_id != user_id and not current_user.is_superuser:
            raise HTTPException(status_code=403, detail="Not authorized")

        comment.deleted_at = datetime.utcnow()
        await session.commit()


# ── Analytics Endpoint ────────────────────────────────────────────

@router.get("/{post_id}/analytics")
async def get_post_analytics(
    post_id: str,
    current_user: User = Depends(get_current_active_user),
):
    """Get view count and comment count for a post."""
    sf = get_async_sessionmaker()
    async with sf() as session:
        post_result = await session.execute(
            select(BlogPostSQL).where(
                BlogPostSQL.id == post_id,
                BlogPostSQL.deleted_at.is_(None),
            )
        )
        post = post_result.scalar_one_or_none()
        if not post:
            raise HTTPException(status_code=404, detail="Post not found")

        comment_count = (
            await session.execute(
                select(func.count(BlogCommentSQL.id)).where(
                    BlogCommentSQL.post_id == post_id,
                    BlogCommentSQL.deleted_at.is_(None),
                )
            )
        ).scalar() or 0

        return {
            "post_id": post.id,
            "title": post.title,
            "view_count": post.view_count or 0,
            "comment_count": comment_count,
            "published_at": post.published_at.isoformat() if post.published_at else None,
        }


# ── Notification Helper ──────────────────────────────────────────

async def _notify_blog_published(post: BlogPostSQL):
    """Send notifications when a blog post is published (news/release only)."""
    try:
        from backend.services.notification_service import notification_service, connection_manager
        from backend.db.mongodb.models.notification import NotificationType

        # Broadcast to all connected WebSocket clients
        await connection_manager.broadcast({
            "type": "blog_published",
            "post_id": post.id,
            "title": post.title,
            "slug": post.slug,
            "category": post.category,
            "author_name": post.author_name,
        })

        logger.info("blog_publish_notification_sent", post_id=post.id, title=post.title)
    except Exception as e:
        logger.error("blog_notification_failed", error=str(e), post_id=post.id)
