from __future__ import annotations

from typing import List

from pydantic import BaseModel
from sqlalchemy import Float, ForeignKey, String, Text, UniqueConstraint
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


# Pydantic Models
class DocumentModel(BaseModel):
    id: int
    title: str
    url: str
    content: str
    token_count: int

    class Config:
        from_attributes = True


class WordModel(BaseModel):
    id: int
    term: str

    class Config:
        from_attributes = True


# SQLAlchemy Table Definitions
class Document(Base):
    __tablename__ = "documents"
    id: Mapped[int] = mapped_column(primary_key=True, index=True)
    title: Mapped[str] = mapped_column(String)
    url: Mapped[str] = mapped_column(String, unique=True)
    content: Mapped[str] = mapped_column(Text)
    token_count: Mapped[int] = mapped_column()

    inverted_indices: Mapped[List["InvertedIndex"]] = relationship(
        back_populates="document"
    )


class Word(Base):
    __tablename__ = "words"
    id: Mapped[int] = mapped_column(primary_key=True, index=True)
    term: Mapped[str] = mapped_column(String, unique=True)

    inverted_indices: Mapped[List["InvertedIndex"]] = relationship(
        back_populates="word"
    )
    doc_frequency: Mapped["DocumentFrequency"] = relationship(back_populates="word")


class InvertedIndex(Base):
    __tablename__ = "inverted_index"
    word_id: Mapped[int] = mapped_column(ForeignKey("words.id"), primary_key=True)
    document_id: Mapped[int] = mapped_column(
        ForeignKey("documents.id"), primary_key=True
    )
    term_frequency: Mapped[int] = mapped_column()

    word: Mapped["Word"] = relationship(back_populates="inverted_indices")
    document: Mapped["Document"] = relationship(back_populates="inverted_indices")

    __table_args__ = (
        UniqueConstraint("word_id", "document_id", name="_word_document_uc"),
    )


class DocumentFrequency(Base):
    __tablename__ = "document_frequency"
    word_id: Mapped[int] = mapped_column(ForeignKey("words.id"), primary_key=True)
    doc_frequency: Mapped[int] = mapped_column()

    word: Mapped["Word"] = relationship(back_populates="doc_frequency")


class SystemStats(Base):
    __tablename__ = "system_stats"
    id: Mapped[int] = mapped_column(primary_key=True)
    total_documents: Mapped[int] = mapped_column()
    average_doc_length: Mapped[float] = mapped_column(Float)
