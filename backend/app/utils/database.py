"""
Database utility functions for the academic publication management system.
"""

import os
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

# Database configuration
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://user:password@localhost/publication_db"
)

# For development/testing, use SQLite
if os.getenv("ENVIRONMENT") == "development" or not DATABASE_URL.startswith("postgresql"):
    DATABASE_URL = "sqlite:///./publication_management.db"
    engine = create_engine(
        DATABASE_URL,
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
        echo=True  # Enable SQL logging for development
    )
else:
    # Production database
    engine = create_engine(DATABASE_URL, pool_pre_ping=True)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


async def init_database():
    """Initialize the database with all tables."""
    # Import all models to ensure they're registered with Base
    from app.models.faculty import Faculty
    from app.models.publication import Publication
    from app.models.plagiarism import PlagiarismScan
    from app.models.export import ExportJob

    # Create all tables
    Base.metadata.create_all(bind=engine)
    print("Database initialized successfully!")


def get_db():
    """Get database session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


async def check_database_connection():
    """Check if database connection is working."""
    try:
        db = SessionLocal()
        # Try to execute a simple query
        db.execute("SELECT 1")
        db.close()
        return True
    except Exception as e:
        print(f"Database connection error: {e}")
        return False


async def create_indexes():
    """Create additional database indexes for performance."""
    from app.models.faculty import Faculty
    from app.models.publication import Publication
    from app.models.plagiarism import PlagiarismScan
    from app.models.export import ExportJob

    # These will be created by SQLAlchemy based on index=True in model definitions
    # Additional indexes can be added here if needed
    pass


async def backup_database():
    """Create a backup of the database (PostgreSQL only)."""
    if DATABASE_URL.startswith("postgresql"):
        import subprocess
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_file = f"backup_publication_db_{timestamp}.sql"

        try:
            cmd = f"pg_dump {DATABASE_URL} > {backup_file}"
            subprocess.run(cmd, shell=True, check=True)
            print(f"Database backup created: {backup_file}")
            return backup_file
        except subprocess.CalledProcessError as e:
            print(f"Backup failed: {e}")
            return None
    else:
        print("Database backup not supported for SQLite in development")
        return None