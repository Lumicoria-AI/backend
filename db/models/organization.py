from datetime import datetime
from typing import Optional
from sqlalchemy import Boolean, Column, DateTime, ForeignKey, Integer, String, Text
from sqlalchemy.orm import relationship

from ..base import Base

class Organization(Base):
    __tablename__ = "organizations"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, unique=True, nullable=False)
    description = Column(Text, nullable=True)
    owner_id = Column(Integer, ForeignKey("users.id", ondelete="SET NULL")) # Creator/Owner of the organization
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    owner = relationship("User", back_populates="owned_organizations")
    members = relationship("UserOrganization", back_populates="organization")
    teams = relationship("Team", back_populates="organization")
    documents = relationship("Document", back_populates="organization") # Documents owned by the organization
    agents = relationship("Agent", back_populates="organization") # Agents owned by the organization

class UserOrganization(Base):
    __tablename__ = "user_organizations"

    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), primary_key=True)
    organization_id = Column(Integer, ForeignKey("organizations.id", ondelete="CASCADE"), primary_key=True)
    role = Column(String, default="member") # e.g., owner, admin, member, guest
    joined_at = Column(DateTime, default=datetime.utcnow)

    user = relationship("User", back_populates="organizations")
    organization = relationship("Organization", back_populates="members")

class Team(Base):
    __tablename__ = "teams"

    id = Column(Integer, primary_key=True, index=True)
    organization_id = Column(Integer, ForeignKey("organizations.id", ondelete="CASCADE"), nullable=False)
    name = Column(String, nullable=False)
    description = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    organization = relationship("Organization", back_populates="teams")
    members = relationship("UserTeam", back_populates="team")
    documents = relationship("Document", back_populates="team") # Documents owned by the team
    agents = relationship("Agent", back_populates="team") # Agents owned by the team

class UserTeam(Base):
    __tablename__ = "user_teams"

    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), primary_key=True)
    team_id = Column(Integer, ForeignKey("teams.id", ondelete="CASCADE"), primary_key=True)
    role = Column(String, default="member") # e.g., leader, member
    joined_at = Column(DateTime, default=datetime.utcnow)

    user = relationship("User", back_populates="teams")
    team = relationship("Team", back_populates="members") 