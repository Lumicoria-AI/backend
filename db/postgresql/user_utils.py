from typing import List, Optional
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select

from ..base import AsyncSessionLocal # Import the session local
from ..models import User, UserProfile, UserSettings # Import the models

async def create_user(*, db: AsyncSession, email: str, firebase_uid: str, hashed_password: Optional[str] = None) -> User:
    """Creates a new user in the database."""
    new_user = User(email=email, firebase_uid=firebase_uid, hashed_password=hashed_password)
    db.add(new_user)
    await db.commit()
    await db.refresh(new_user)
    return new_user

async def get_user_by_email(*, db: AsyncSession, email: str) -> Optional[User]:
    """Retrieves a user by email."""
    result = await db.execute(select(User).where(User.email == email))
    return result.scalars().first()

async def get_user_by_firebase_uid(*, db: AsyncSession, firebase_uid: str) -> Optional[User]:
    """Retrieves a user by Firebase UID."""
    result = await db.execute(select(User).where(User.firebase_uid == firebase_uid))
    return result.scalars().first()

async def get_user_by_id(*, db: AsyncSession, user_id: int) -> Optional[User]:
    """Retrieves a user by ID."""
    result = await db.execute(select(User).where(User.id == user_id))
    return result.scalars().first()

async def get_all_users(*, db: AsyncSession, skip: int = 0, limit: int = 100) -> List[User]:
    """Retrieves all users with pagination."""
    result = await db.execute(select(User).offset(skip).limit(limit))
    return result.scalars().all()

async def update_user(*, db: AsyncSession, user: User, **kwargs) -> User:
    """Updates user information."""
    for key, value in kwargs.items():
        setattr(user, key, value)
    db.add(user)
    await db.commit()
    await db.refresh(user)
    return user

async def delete_user(*, db: AsyncSession, user: User) -> None:
    """Deletes a user from the database."""
    await db.delete(user)
    await db.commit() 