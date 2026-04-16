"""
AI Blog Writer Service — generates and saves blog posts via LLM.
Supports internal triggers (API, Celery tasks, agent-to-agent).
"""

import json
import re
import random
import string
from datetime import datetime
from typing import Optional

import bleach
from bleach.css_sanitizer import CSSSanitizer
import structlog

from backend.db.postgres import get_async_sessionmaker
from backend.db.postgres_models import BlogPostSQL, BlogPostStatus, AuthorType

logger = structlog.get_logger(__name__)

# Same sanitization config as blog.py
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

BLOG_WRITER_SYSTEM_PROMPT = """You are a professional blog writer for Lumicoria AI, an AI agent platform.
Write engaging, well-structured blog posts in clean HTML suitable for a TipTap rich text editor.

Output ONLY valid JSON with these fields:
{
  "title": "Post title",
  "subtitle": "A brief subtitle",
  "excerpt": "1-2 sentence summary for the blog listing card",
  "content": "<h2>...</h2><p>...</p>...",
  "tags": ["tag1", "tag2", "tag3"]
}

HTML guidelines:
- Use <h2> and <h3> for section headings (never <h1>, the title is separate)
- Use <p> for paragraphs, <strong> for emphasis, <ul>/<ol> for lists
- Use <blockquote> for callouts or key takeaways
- Use <code> for inline code, <pre><code> for code blocks
- Keep the tone professional but approachable
- Aim for 800-1500 words
- Do NOT include the title in the HTML content — it's rendered separately"""


def _slugify(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^\w\s-]", "", text)
    text = re.sub(r"[\s_]+", "-", text)
    text = re.sub(r"-+", "-", text).strip("-")
    return text[:200]


def _random_suffix(length: int = 4) -> str:
    return "".join(random.choices(string.ascii_lowercase + string.digits, k=length))


class BlogWriterService:
    """Generates blog posts via LLM and saves them to Postgres."""

    async def generate_and_publish(
        self,
        topic: str,
        category: str = "engineering",
        auto_publish: bool = False,
        cover_image_url: Optional[str] = None,
    ) -> BlogPostSQL:
        """Generate a blog post from a topic and save it."""
        from backend.ai_models.registry import get_llm_client
        from backend.ai_models.base import LLMMessage, MessageRole, LLMConfig

        client = get_llm_client(provider="gemini")

        messages = [
            LLMMessage(role=MessageRole.SYSTEM, content=BLOG_WRITER_SYSTEM_PROMPT),
            LLMMessage(
                role=MessageRole.USER,
                content=f"Write a blog post about: {topic}\n\nCategory: {category}",
            ),
        ]

        config = LLMConfig(temperature=0.7, max_tokens=4096)
        response = await client.generate(messages, config=config)

        # Parse LLM JSON output
        raw = response.content.strip()
        # Strip markdown code fences if present
        if raw.startswith("```"):
            raw = re.sub(r"^```(?:json)?\s*", "", raw)
            raw = re.sub(r"\s*```$", "", raw)

        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            logger.error("ai_blog_json_parse_failed", raw_preview=raw[:500])
            raise ValueError("AI returned invalid JSON — retry or adjust the topic.")

        title = data.get("title", topic)
        subtitle = data.get("subtitle")
        excerpt = data.get("excerpt")
        content_html = data.get("content", "")
        tags = data.get("tags", [])

        # Sanitize
        _css_sanitizer = CSSSanitizer(allowed_css_properties=["color", "font-weight", "text-align", "margin", "padding"])
        content_html = bleach.clean(
            content_html, tags=ALLOWED_TAGS, attributes=ALLOWED_ATTRS,
            css_sanitizer=_css_sanitizer, strip=True,
        )

        # Generate slug
        slug = f"{_slugify(title)}-{datetime.utcnow().strftime('%Y%m%d')}"

        sf = get_async_sessionmaker()
        async with sf() as session:
            from sqlalchemy import select

            existing = await session.execute(
                select(BlogPostSQL.id).where(BlogPostSQL.slug == slug)
            )
            if existing.scalar_one_or_none():
                slug = f"{slug}-{_random_suffix()}"

            now = datetime.utcnow()
            status = BlogPostStatus.PUBLISHED if auto_publish else BlogPostStatus.DRAFT

            post = BlogPostSQL(
                slug=slug,
                title=title,
                subtitle=subtitle,
                content=content_html,
                excerpt=excerpt,
                author_id="ai-agent",
                author_type=AuthorType.AI_AGENT,
                author_name="Lumicoria AI",
                author_avatar_url=None,
                author_title="AI Content Writer",
                cover_image_url=cover_image_url,
                category=category,
                tags=tags[:10],  # Cap at 10 tags
                status=status,
                featured=False,
                published_at=now if auto_publish else None,
                created_at=now,
                updated_at=now,
            )
            session.add(post)
            await session.commit()
            await session.refresh(post)

            logger.info(
                "ai_blog_post_created",
                post_id=post.id,
                title=title,
                status=status.value,
                slug=slug,
            )

            # Notify if auto-published news/release
            if auto_publish and category in ("news", "release"):
                try:
                    from backend.api.v1.endpoints.blog import _notify_blog_published
                    await _notify_blog_published(post)
                except Exception as e:
                    logger.error("ai_blog_notification_failed", error=str(e))

            return post


blog_writer_service = BlogWriterService()
