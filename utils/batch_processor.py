# utils/batch_processor.py
"""
Batch Processing Utilities
"""

import asyncio
import logging
from typing import List, Callable, Any, Dict
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import time
from pathlib import Path

class BatchProcessor:
    """Handles batch processing of files and operations"""
    
    def __init__(self, max_workers: int = 4, gpu_count: int = 3):
        self.max_workers = max_workers
        self.gpu_count = gpu_count
        self.logger = logging.getLogger(__name__)
        
        # Thread and process pools
        self.thread_pool = ThreadPoolExecutor(max_workers=max_workers)
        self.process_pool = ProcessPoolExecutor(max_workers=max_workers)
    
    async def process_files_batch(self, file_paths: List[Path], 
                                processor_func: Callable, 
                                batch_size: int = 10,
                                use_gpu_distribution: bool = True) -> List[Any]:
        """Process files in batches with GPU distribution"""
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
                
                # Distribute batch across GPUs if enabled
                if use_gpu_distribution:
                    batch_results = await self._process_batch_with_gpu_distribution(
                        batch, processor_func, batch_idx
                    )
                else:
                    batch_results = await self._process_batch_sequential(
                        batch, processor_func
                    )
                
                results.extend(batch_results)
                
                batch_time = time.time() - batch_start_time
                self.logger.info(
                    f"Batch {batch_idx + 1}/{len(batches)} completed in {batch_time:.2f}s"
                )
            
            return results
            
        except Exception as e:
            self.logger.error(f"Batch processing failed: {str(e)}")
            raise
    
    async def _process_batch_with_gpu_distribution(self, batch: List[Path], 
                                                 processor_func: Callable,
                                                 batch_idx: int) -> List[Any]:
        """Process batch with GPU distribution"""
        tasks = []
        
        for i, file_path in enumerate(batch):
            # Assign GPU based on file index
            gpu_id = i % self.gpu_count
            
            # Create async task
            task = asyncio.create_task(
                self._process_file_on_gpu(file_path, processor_func, gpu_id)
            )
            tasks.append(task)
        
        # Wait for all tasks to complete
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        return results
    
    async def _process_batch_sequential(self, batch: List[Path], 
                                      processor_func: Callable) -> List[Any]:
        """Process batch sequentially"""
        results = []
        
        for file_path in batch:
            try:
                result = await processor_func(file_path)
                results.append(result)
            except Exception as e:
                self.logger.error(f"Failed to process {file_path}: {str(e)}")
                results.append({"error": str(e), "file": str(file_path)})
        
        return results
    
    async def _process_file_on_gpu(self, file_path: Path, processor_func: Callable, gpu_id: int) -> Any:
        """Process single file on specific GPU"""
        try:
            # Set GPU device context if using CUDA
            import torch
            if torch.cuda.is_available():
                with torch.cuda.device(gpu_id):
                    result = await processor_func(file_path)
            else:
                result = await processor_func(file_path)
            
            return result
            
        except Exception as e:
            self.logger.error(f"GPU {gpu_id} processing failed for {file_path}: {str(e)}")
            return {"error": str(e), "file": str(file_path), "gpu_id": gpu_id}
    
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
            
            # Collect results
            results = []
            for future in futures:
                batch_results = future.result(timeout=300)  # 5 minute timeout
                results.extend(batch_results)
            
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
    
    def shutdown(self):
        """Shutdown thread and process pools"""
        self.thread_pool.shutdown(wait=True)
        self.process_pool.shutdown(wait=True)
        self.logger.info("Batch processor shutdown complete")
