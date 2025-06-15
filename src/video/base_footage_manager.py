"""
Base Footage Manager - Core functionality and initialization.
"""

import json
import time
from pathlib import Path
from typing import Optional, Dict
from loguru import logger


class BaseFootageManager:
    """
    Base class for footage management with core functionality.
    """
    
    def __init__(self, storage_dir: Optional[Path] = None):
        """Initialize the base footage manager."""
        self.storage_dir = storage_dir or Path(__file__).parent / "data" / "footage"
        self.raw_footage_dir = self.storage_dir / "raw"
        self.processed_footage_dir = self.storage_dir / "processed"
        self.metadata_file = self.storage_dir / "footage_metadata.json"
        
        # Create directories
        for dir_path in [self.storage_dir, self.raw_footage_dir, self.processed_footage_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Load metadata
        self.metadata = self._load_metadata()
        
        logger.info(f"BaseFootageManager initialized with storage: {self.storage_dir}")
    
    def _load_metadata(self) -> Dict:
        """Load footage metadata from file."""
        try:
            if self.metadata_file.exists():
                with open(self.metadata_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load metadata: {e}")
        
        return {
            "sources": {},
            "videos": {},
            "categories": {
                "high_action": [],
                "medium_action": [],
                "ambient": []
            },
            "last_updated": None,
            "total_duration": 0,
            "total_videos": 0
        }
    
    def _save_metadata(self):
        """Save metadata to file."""
        try:
            self.metadata["last_updated"] = time.time()
            with open(self.metadata_file, 'w', encoding='utf-8') as f:
                json.dump(self.metadata, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Failed to save metadata: {e}")
    
    def get_storage_info(self) -> Dict:
        """Get storage information and statistics."""
        try:
            raw_size = sum(f.stat().st_size for f in self.raw_footage_dir.glob("*") if f.is_file())
            processed_size = sum(f.stat().st_size for f in self.processed_footage_dir.glob("*") if f.is_file())
            
            return {
                "raw_footage_mb": raw_size / (1024 * 1024),
                "processed_footage_mb": processed_size / (1024 * 1024),
                "total_videos": self.metadata.get("total_videos", 0),
                "categories": {k: len(v) for k, v in self.metadata.get("categories", {}).items()}
            }
        except Exception as e:
            logger.error(f"Failed to get storage info: {e}")
            return {}
