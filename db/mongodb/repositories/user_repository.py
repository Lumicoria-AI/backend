from typing import Optional, List, Dict, Any, Type, TypeVar, Union
from pymongo import ASCENDING
from backend.db.mongodb.base_repository import BaseRepository
from backend.models.user import UserCreateOAuth, UserProfile, UserSettings
from backend.db.mongodb.models.user import UserInDB, UserCreate, UserUpdate
import structlog
from datetime import datetime
from bson import ObjectId
from motor.motor_asyncio import AsyncIOMotorDatabase, AsyncIOMotorCollection
from backend.models.mongodb_models import MongoBaseModel
from ..mongodb import MongoDB
from pymongo import UpdateOne
from fastapi import UploadFile
import os
import uuid
from backend.core.config import settings

logger = structlog.get_logger()

T = TypeVar('T', bound=MongoBaseModel)

class UserRepository(BaseRepository[UserInDB]):
    def __init__(
        self,
        collection: AsyncIOMotorCollection,
        model_class: Type[UserInDB],
        profile_collection: AsyncIOMotorCollection,
        settings_collection: AsyncIOMotorCollection
    ):
        super().__init__(collection, model_class)
        self.profile_collection = profile_collection
        self.settings_collection = settings_collection

    async def _create_indexes(self):
        await super()._create_indexes()
        await self.collection.create_index("email", unique=True)
        await self.collection.create_index("created_at")
        await self.collection.create_index("organization_ids")
        await self.collection.create_index("role_ids")
        await self.collection.create_index([("full_name", ASCENDING), ("email", ASCENDING)])
        
        await self.profile_collection.create_index("user_id", unique=True)
        await self.settings_collection.create_index("user_id", unique=True)

    async def create_user(self, user: UserCreate) -> UserInDB:
        """Create a new user with a password."""
        # Access collection directly
        user_dict = user.model_dump(exclude={"password"})
        user_dict["created_at"] = datetime.utcnow()
        
        # Hash the password if not already hashed
        if user.hashed_password:
            user_dict["hashed_password"] = user.hashed_password
        else:
            from backend.core.security import get_password_hash
            user_dict["hashed_password"] = get_password_hash(user.password)
            
        # Ensure firebase_uid is set even if None initially
        user_dict["firebase_uid"] = user_dict.get("firebase_uid") # Explicitly get or set None
        
        # Use self.collection explicitly
        result = await self.collection.insert_one(user_dict)
        # Use self.collection explicitly for find_one
        created_doc = await self.collection.find_one({"_id": result.inserted_id})
        if created_doc:
             return self.model_class(**created_doc)
        # Handle case where insert_one succeeds but find_one fails immediately after
        raise Exception("Failed to retrieve user after creation")

    async def create_user_oauth(self, user: UserCreateOAuth) -> UserInDB:
        """Create a new user from OAuth provider (no password)."""
        user_dict = user.model_dump()
        user_dict["created_at"] = datetime.utcnow()
        # Store a placeholder for hashed_password for OAuth users
        user_dict["hashed_password"] = "OAUTH_USER"
        # firebase_uid is required for UserCreateOAuth and will be in user_dict
        
        # Use self.collection explicitly
        result = await self.collection.insert_one(user_dict)
        # Use self.collection explicitly for find_one
        created_doc = await self.collection.find_one({"_id": result.inserted_id})
        if created_doc:
             return self.model_class(**created_doc)
        # Handle case where insert_one succeeds but find_one fails immediately after
        raise Exception("Failed to retrieve OAuth user after creation")

    async def get_user_by_email(self, email: str) -> Optional[UserInDB]:
        user_dict = await self.collection.find_one({"email": email})
        if user_dict:
            return UserInDB(**user_dict)
        return None

    async def get_user_by_id(self, user_id: str) -> Optional[UserInDB]:
        try:
            user_dict = await self.collection.find_one({"_id": ObjectId(user_id)})
            if user_dict:
                return UserInDB(**user_dict)
        except:
            pass
        return None

    async def get_user_by_firebase_uid(self, firebase_uid: str) -> Optional[UserInDB]:
        user_dict = await self.collection.find_one({"firebase_uid": firebase_uid})
        if user_dict:
            return UserInDB(**user_dict)
        return None    
    async def update_user(self, user_id: str, user_update: Union[UserUpdate, dict]) -> Optional[UserInDB]:
        if isinstance(user_update, dict):
            update_data = user_update
        else:
            update_data = user_update.model_dump(exclude_unset=True)
            
        if update_data:
            # Add updated timestamp
            update_data["updated_at"] = datetime.utcnow()
            
            # Special handling for onboarding_completed field
            if "onboarding_completed" in update_data and update_data["onboarding_completed"] is True:
                logger.info("Setting onboarding_completed to True for user", user_id=user_id)
                # Ensure onboarding_completed_at is also set
                if "onboarding_completed_at" not in update_data:
                    update_data["onboarding_completed_at"] = datetime.utcnow()
            
            # Perform the update
            result = await self.collection.find_one_and_update(
                {"_id": ObjectId(user_id)},
                {"$set": update_data},
                return_document=True
            )
            
            if result:
                # Make sure we properly construct the UserInDB object
                user = UserInDB(**result)
                
                # Log confirmation of onboarding status
                if "onboarding_completed" in update_data:
                    logger.info("User updated with onboarding status", 
                               user_id=user_id, 
                               onboarding_completed=getattr(user, "onboarding_completed", False))
                return user
        return None

    async def create_user_profile(self, user_id: str, profile: UserProfile) -> UserProfile:
        profile_dict = profile.model_dump()
        profile_dict["user_id"] = ObjectId(user_id)
        profile_dict["created_at"] = datetime.utcnow()
        result = await self.profile_collection.insert_one(profile_dict)
        profile_dict["_id"] = result.inserted_id
        created_profile_dict = await self.profile_collection.find_one({"_id": result.inserted_id})
        if created_profile_dict:
             return UserProfile(**created_profile_dict)
        raise Exception("Failed to retrieve user profile after creation")

    async def get_user_profile(self, user_id: str) -> Optional[UserProfile]:
        profile_dict = await self.profile_collection.find_one({"user_id": ObjectId(user_id)})
        if profile_dict:
            return UserProfile(**profile_dict)
        return None

    async def create_user_settings(self, user_id: str, settings: UserSettings) -> UserSettings:
        settings_dict = settings.model_dump()
        settings_dict["user_id"] = ObjectId(user_id)
        settings_dict["created_at"] = datetime.utcnow()
        result = await self.settings_collection.insert_one(settings_dict)
        settings_dict["_id"] = result.inserted_id
        created_settings_dict = await self.settings_collection.find_one({"_id": result.inserted_id})
        if created_settings_dict:
            return UserSettings(**created_settings_dict)
        raise Exception("Failed to retrieve user settings after creation")

    async def get_user_settings(self, user_id: str) -> Optional[UserSettings]:
        settings_dict = await self.settings_collection.find_one({"user_id": ObjectId(user_id)})
        if settings_dict:
            settings_dict["user_id"] = str(settings_dict["user_id"])
            if "_id" in settings_dict:
                settings_dict["_id"] = str(settings_dict["_id"])
            return UserSettings(**settings_dict)
        return None

    async def update_user_settings(self, user_id: str, settings_data: dict) -> Optional[UserSettings]:
        """Update user settings, creating the document if it doesn't exist."""
        settings_data["updated_at"] = datetime.utcnow()
        result = await self.settings_collection.find_one_and_update(
            {"user_id": ObjectId(user_id)},
            {"$set": settings_data},
            return_document=True,
        )
        if not result:
            # No existing settings doc — create one
            settings_data["user_id"] = ObjectId(user_id)
            settings_data["created_at"] = datetime.utcnow()
            insert_result = await self.settings_collection.insert_one(settings_data)
            result = await self.settings_collection.find_one({"_id": insert_result.inserted_id})
        if result:
            # Convert ObjectId fields to str before constructing Pydantic model
            result["user_id"] = str(result["user_id"])
            if "_id" in result:
                result["_id"] = str(result["_id"])
            return UserSettings(**result)
        return None

    async def get_users_by_organization(self, organization_id: str, skip: int = 0, limit: int = 100) -> List[UserInDB]:
        return await self.find_many(
            {"organization_ids": organization_id},
            skip=skip,
            limit=limit,
            sort=[("created_at", ASCENDING)]
        )

    async def get_users_by_role(self, role_id: str, skip: int = 0, limit: int = 100) -> List[UserInDB]:
        return await self.find_many(
            {"role_ids": role_id},
            skip=skip,
            limit=limit,
            sort=[("created_at", ASCENDING)]
        )

    async def add_to_organization(self, user_id: str, organization_id: str) -> Optional[UserInDB]:
        return await self.update(
            user_id,
            {"$addToSet": {"organization_ids": organization_id}}
        )

    async def remove_from_organization(self, user_id: str, organization_id: str) -> Optional[UserInDB]:
        return await self.update(
            user_id,
            {"$pull": {"organization_ids": organization_id}}
        )

    async def add_role(self, user_id: str, role_id: str) -> Optional[UserInDB]:
        return await self.update(
            user_id,
            {"$addToSet": {"role_ids": role_id}}
        )

    async def remove_role(self, user_id: str, role_id: str) -> Optional[UserInDB]:
        return await self.update(
            user_id,
            {"$pull": {"role_ids": role_id}}
        )

    async def search_users(
        self,
        query: str,
        organization_id: Optional[str] = None,
        skip: int = 0,
        limit: int = 100
    ) -> List[UserInDB]:
        search_filter = {
            "$or": [
                {"full_name": {"$regex": query, "$options": "i"}},
                {"email": {"$regex": query, "$options": "i"}},
                {"profile.job_title": {"$regex": query, "$options": "i"}},
                {"profile.company": {"$regex": query, "$options": "i"}}
            ]
        }
        
        if organization_id:
            search_filter["organization_ids"] = organization_id

        return await self.find_many(
            search_filter,
            skip=skip,
            limit=limit,
            sort=[("created_at", ASCENDING)]
        )

    async def get_active_users(self, skip: int = 0, limit: int = 100) -> List[UserInDB]:
        return await self.find_many(
            {"is_active": True},
            skip=skip,
            limit=limit,
            sort=[("created_at", ASCENDING)]
        )    
    async def get_superusers(self, skip: int = 0, limit: int = 100) -> List[UserInDB]:
        return await self.find_many(
            {"is_superuser": True},
            skip=skip,
            limit=limit,
            sort=[("created_at", ASCENDING)]
        )
        
    async def upload_avatar(self, user_id: str, file: UploadFile) -> str:
        """Upload user avatar and return the URL.
        
        Args:
            user_id: The ID of the user
            file: The uploaded file
            
        Returns:
            The URL of the uploaded avatar
        """
        # Create uploads directory if it doesn't exist 
        upload_dir = os.path.join(settings.UPLOAD_DIR, "avatars")
        os.makedirs(upload_dir, exist_ok=True)
        
        # Generate unique filename
        file_ext = os.path.splitext(file.filename)[1]
        unique_filename = f"{user_id}_{uuid.uuid4()}{file_ext}"
        file_path = os.path.join(upload_dir, unique_filename)
        
        # Save file
        contents = await file.read()
        with open(file_path, "wb") as f:
            f.write(contents)
        
        # Generate URL for the avatar - This path will be served by the static files middleware
        # Ensure the URL starts with / to make it relative to the server root
        avatar_url = f"/uploads/avatars/{unique_filename}"
        
        logger.info("Avatar uploaded", 
                   user_id=user_id, 
                   file_path=file_path, 
                   url=avatar_url, 
                   file_size=len(contents),
                   content_type=file.content_type)
                   
        return avatar_url

# Create a singleton instance (remains None initially, managed by dependency)
user_repository: Optional[UserRepository] = None

async def get_user_repository() -> UserRepository:
    global user_repository
    if user_repository is None:
        try:
            logger.info("Initializing user repository")
            user_collection = await MongoDB.get_collection("users")
            user_profile_collection = await MongoDB.get_collection("user_profiles")
            user_settings_collection = await MongoDB.get_collection("user_settings")
            user_repository = UserRepository(
                user_collection,
                UserInDB,
                user_profile_collection,
                user_settings_collection
            )

            await user_repository._create_indexes()
            logger.info("User repository initialized successfully")
        except Exception as e:
            logger.error("Failed to initialize user repository", error=str(e), exc_info=True)
            # Re-raise the exception to prevent silent failures
            raise

    return user_repository