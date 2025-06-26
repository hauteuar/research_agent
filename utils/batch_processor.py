# utils/batch_processor.py - FIXED VERSION
"""
Batch Processing Utilities with Enhanced GPU Management and Error Handling
"""

import asyncio
import logging
from typing import List, Callable, Any, Dict
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import time
from pathlib import Path

class BatchProcessor:
    """Handles batch processing of files and operations with improved GPU management"""
    
    def __init__(self, max_workers: int = 4, gpu_count: int = 4):  # Changed to 4 GPUs
        self.max_workers = max_workers
        self.gpu_count = gpu_count
        self.logger = logging.getLogger(__name__)
        
        # Thread and process pools
        self.thread_pool = ThreadPoolExecutor(max_workers=max_workers)
        self.process_pool = ProcessPoolExecutor(max_workers=max_workers)
        
        # Track active tasks per GPU
        self.gpu_task_count = {i: 0 for i in range(gpu_count)}
        self.gpu_lock = asyncio.Lock()
    
    async def process_files_batch(self, file_paths: List[Path], 
                                processor_func: Callable, 
                                batch_size: int = 10,
                                use_gpu_distribution: bool = True) -> List[Any]:
        """Process files in batches with intelligent GPU distribution"""
        results = []
        
        try:
            # Split files into batches
            batches = [
                file_paths[i:i + batch_size] 
                for i in range(0, len(file_paths), batch_size)
            ]
            
            self.logger.info(f"Processing {len(file_paths)} files in {len(batches)} batches")
            
            for batch_idx, batch in enumerate(batches):
                batch_start_time = time.time()
                
                # FIX 1: Better GPU distribution strategy
                if use_gpu_distribution:
                    batch_results = await self._process_batch_with_smart_gpu_distribution(
                        batch, processor_func, batch_idx
                    )
                else:
                    batch_results = await self._process_batch_sequential(
                        batch, processor_func
                    )
                
                # FIX 2: Filter out None results and handle errors properly
                valid_results = [r for r in batch_results if r is not None]
                results.extend(valid_results)
                
                batch_time = time.time() - batch_start_time
                success_count = len([r for r in valid_results if isinstance(r, dict) and r.get('status') == 'success'])
                
                self.logger.info(
                    f"Batch {batch_idx + 1}/{len(batches)} completed in {batch_time:.2f}s - "
                    f"{success_count}/{len(batch)} files successful"
                )
            
            return results
            
        except Exception as e:
            self.logger.error(f"Batch processing failed: {str(e)}")
            raise
    
    async def _process_batch_with_smart_gpu_distribution(self, batch: List[Path], 
                                                       processor_func: Callable,
                                                       batch_idx: int) -> List[Any]:
        """Process batch with intelligent GPU distribution"""
        
        # FIX 3: Don't force GPU assignment in batch processor
        # Let the coordinator handle GPU allocation through the processor_func
        
        # Create tasks with controlled concurrency
        semaphore = asyncio.Semaphore(self.max_workers)
        
        async def process_single_file(file_path: Path, file_idx: int):
            async with semaphore:
                try:
                    # FIX 4: Let the processor_func handle GPU allocation
                    # Don't try to force GPU context here
                    result = await processor_func(file_path)
                    
                    if result is None:
                        self.logger.warning(f"Processor returned None for {file_path}")
                        return {"status": "error", "file": str(file_path), "error": "Processor returned None"}
                    
                    return result
                    
                except Exception as e:
                    self.logger.error(f"Error processing {file_path}: {str(e)}")
                    return {"status": "error", "file": str(file_path), "error": str(e)}
        
        # Create tasks for all files in batch
        tasks = [
            asyncio.create_task(process_single_file(file_path, i))
            for i, file_path in enumerate(batch)
        ]
        
        # Wait for all tasks with timeout
        try:
            results = await asyncio.wait_for(
                asyncio.gather(*tasks, return_exceptions=True),
                timeout=300  # 5 minute timeout per batch
            )
            
            # FIX 5: Handle exceptions properly
            processed_results = []
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    self.logger.error(f"Task {i} failed with exception: {result}")
                    processed_results.append({
                        "status": "error", 
                        "file": str(batch[i]), 
                        "error": str(result)
                    })
                else:
                    processed_results.append(result)
            
            return processed_results
            
        except asyncio.TimeoutError:
            self.logger.error(f"Batch {batch_idx} timed out after 5 minutes")
            # Cancel remaining tasks
            for task in tasks:
                if not task.done():
                    task.cancel()
            
            return [{"status": "error", "file": str(fp), "error": "Timeout"} for fp in batch]
    
    async def _process_batch_sequential(self, batch: List[Path], 
                                      processor_func: Callable) -> List[Any]:
        """Process batch sequentially with better error handling"""
        results = []
        
        for i, file_path in enumerate(batch):
            try:
                self.logger.debug(f"Processing file {i+1}/{len(batch)}: {file_path.name}")
                result = await processor_func(file_path)
                
                if result is None:
                    result = {"status": "error", "file": str(file_path), "error": "Processor returned None"}
                
                results.append(result)
                
            except Exception as e:
                self.logger.error(f"Failed to process {file_path}: {str(e)}")
                results.append({"status": "error", "file": str(file_path), "error": str(e)})
        
        return results
    
    # FIX 6: Remove the problematic _process_file_on_gpu method
    # GPU allocation should be handled by the coordinator, not the batch processor
    
    def process_cpu_intensive_batch(self, items: List[Any], 
                                  processor_func: Callable,
                                  batch_size: int = None) -> List[Any]:
        """Process CPU-intensive tasks using process pool"""
        batch_size = batch_size or self.max_workers
        
        try:
            # Submit tasks to process pool
            futures = []
            
            for i in range(0, len(items), batch_size):
                batch = items[i:i + batch_size]
                future = self.process_pool.submit(self._process_cpu_batch, batch, processor_func)
                futures.append(future)
            
            # Collect results with timeout
            results = []
            for future in futures:
                try:
                    batch_results = future.result(timeout=300)  # 5 minute timeout
                    results.extend(batch_results)
                except Exception as e:
                    self.logger.error(f"CPU batch processing failed: {str(e)}")
                    # Add error results for this batch
                    results.extend([{"error": str(e)} for _ in range(batch_size)])
            
            return results
            
        except Exception as e:
            self.logger.error(f"CPU batch processing failed: {str(e)}")
            raise
    
    @staticmethod
    def _process_cpu_batch(batch: List[Any], processor_func: Callable) -> List[Any]:
        """Process batch in separate process"""
        results = []
        
        for item in batch:
            try:
                result = processor_func(item)
                results.append(result)
            except Exception as e:
                results.append({"error": str(e), "item": str(item)})
        
        return results
    
    def get_batch_statistics(self) -> Dict[str, Any]:
        """Get current batch processing statistics"""
        return {
            "max_workers": self.max_workers,
            "gpu_count": self.gpu_count,
            "thread_pool_active": self.thread_pool._threads,
            "process_pool_active": len(self.process_pool._processes) if hasattr(self.process_pool, '_processes') else 0
        }
    
    def shutdown(self):
        """Shutdown thread and process pools"""
        try:
            self.thread_pool.shutdown(wait=True)
            self.process_pool.shutdown(wait=True)
            self.logger.info("Batch processor shutdown complete")
        except Exception as e:
            self.logger.error(f"Error during batch processor shutdown: {str(e)}")