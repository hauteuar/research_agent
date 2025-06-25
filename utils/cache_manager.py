# utils/cache_manager.py
"""
Cache Manager for efficient data retrieval
"""

import time
import threading
import json
import hashlib
from typing import Any, Optional, Dict
from collections import OrderedDict
import pickle
import logging

class CacheManager:
    """Thread-safe LRU cache with TTL support"""
    
    def __init__(self, max_size: int = 1000, default_ttl: int = 3600):
        self.max_size = max_size
        self.default_ttl = default_ttl
        self.cache = OrderedDict()
        self.timestamps = {}
        self.ttls = {}
        self.lock = threading.RLock()
        self.logger = logging.getLogger(__name__)
        
        # Statistics
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        
        # Start cleanup thread
        self.cleanup_thread = threading.Thread(target=self._cleanup_expired, daemon=True)
        self.cleanup_thread.start()
    
    def _generate_key(self, key: str) -> str:
        """Generate consistent cache key"""
        if isinstance(key, str):
            return hashlib.md5(key.encode()).hexdigest()
        else:
            return hashlib.md5(str(key).encode()).hexdigest()
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        with self.lock:
            cache_key = self._generate_key(key)
            
            if cache_key not in self.cache:
                self.misses += 1
                return None
            
            # Check if expired
            if self._is_expired(cache_key):
                self._remove_key(cache_key)
                self.misses += 1
                return None
            
            # Move to end (mark as recently used)
            value = self.cache.pop(cache_key)
            self.cache[cache_key] = value
            self.hits += 1
            
            return value
    
    def set(self, key: str, value: Any, ttl: int = None) -> bool:
        """Set value in cache"""
        with self.lock:
            cache_key = self._generate_key(key)
            ttl = ttl or self.default_ttl
            
            # Remove existing key if present
            if cache_key in self.cache:
                self._remove_key(cache_key)
            
            # Check size limit
            while len(self.cache) >= self.max_size:
                self._evict_lru()
            
            # Add new item
            self.cache[cache_key] = value
            self.timestamps[cache_key] = time.time()
            self.ttls[cache_key] = ttl
            
            return True
    
    def delete(self, key: str) -> bool:
        """Delete key from cache"""
        with self.lock:
            cache_key = self._generate_key(key)
            
            if cache_key in self.cache:
                self._remove_key(cache_key)
                return True
            
            return False
    
    def clear(self):
        """Clear all cache entries"""
        with self.lock:
            self.cache.clear()
            self.timestamps.clear()
            self.ttls.clear()
            self.logger.info("Cache cleared")
    
    def _is_expired(self, cache_key: str) -> bool:
        """Check if cache entry is expired"""
        if cache_key not in self.timestamps:
            return True
        
        age = time.time() - self.timestamps[cache_key]
        ttl = self.ttls.get(cache_key, self.default_ttl)
        
        return age > ttl
    
    def _remove_key(self, cache_key: str):
        """Remove key and associated metadata"""
        if cache_key in self.cache:
            del self.cache[cache_key]
        if cache_key in self.timestamps:
            del self.timestamps[cache_key]
        if cache_key in self.ttls:
            del self.ttls[cache_key]
    
    def _evict_lru(self):
        """Evict least recently used item"""
        if self.cache:
            lru_key = next(iter(self.cache))
            self._remove_key(lru_key)
            self.evictions += 1
    
    def _cleanup_expired(self):
        """Background thread to clean up expired entries"""
        while True:
            try:
                with self.lock:
                    expired_keys = [
                        key for key in self.cache.keys()
                        if self._is_expired(key)
                    ]
                    
                    for key in expired_keys:
                        self._remove_key(key)
                    
                    if expired_keys:
                        self.logger.debug(f"Cleaned up {len(expired_keys)} expired cache entries")
                
                time.sleep(300)  # Clean up every 5 minutes
                
            except Exception as e:
                self.logger.error(f"Cache cleanup error: {str(e)}")
                time.sleep(300)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        with self.lock:
            total_requests = self.hits + self.misses
            hit_rate = (self.hits / total_requests) * 100 if total_requests > 0 else 0
            
            return {
                "size": len(self.cache),
                "max_size": self.max_size,
                "hits": self.hits,
                "misses": self.misses,
                "hit_rate": hit_rate,
                "evictions": self.evictions,
                "memory_usage_mb": self._estimate_memory_usage() / (1024 * 1024)
            }
    
    def _estimate_memory_usage(self) -> int:
        """Estimate cache memory usage in bytes"""
        try:
            total_size = 0
            for key, value in self.cache.items():
                # Rough estimation using pickle
                total_size += len(pickle.dumps(value))
                total_size += len(key.encode())
            
            return total_size
        except:
            return 0
    
    def get_keys_by_pattern(self, pattern: str) -> List[str]:
        """Get keys matching a pattern"""
        with self.lock:
            import re
            pattern_regex = re.compile(pattern)
            
            matching_keys = []
            for key in self.cache.keys():
                if pattern_regex.search(key):
                    matching_keys.append(key)
            
            return matching_keys
    
    def cache_info(self) -> str:
        """Get formatted cache information"""
        stats = self.get_stats()
        
        info = f"""
Cache Information:
- Size: {stats['size']}/{stats['max_size']} entries
- Hit Rate: {stats['hit_rate']:.2f}%
- Hits: {stats['hits']}, Misses: {stats['misses']}
- Evictions: {stats['evictions']}
- Memory Usage: {stats['memory_usage_mb']:.2f} MB
        """
        
        return info.strip()
