"""
Memory Monitoring and Management Utilities

This module provides memory monitoring and cleanup utilities to prevent
server timeouts and disconnections in resource-constrained environments.
"""

import gc
import psutil
import logging
import torch
from typing import Optional, Dict, Any
from contextlib import contextmanager
import time

logger = logging.getLogger(__name__)


class MemoryMonitor:
    """
    Memory monitoring and management utility.
    """
    
    def __init__(self, memory_limit: float = 0.8, cleanup_frequency: int = 10):
        """
        Initialize memory monitor.
        
        Args:
            memory_limit: Maximum memory usage ratio (0.0 to 1.0)
            cleanup_frequency: Frequency of automatic cleanup (steps)
        """
        self.memory_limit = memory_limit
        self.cleanup_frequency = cleanup_frequency
        self.total_memory = psutil.virtual_memory().total
        self.memory_threshold = self.total_memory * self.memory_limit
        self.step_count = 0
        
        logger.info(f"Memory monitor initialized:")
        logger.info(f"  Total memory: {self.total_memory / (1024**3):.2f} GB")
        logger.info(f"  Memory limit: {self.memory_limit * 100:.1f}%")
        logger.info(f"  Memory threshold: {self.memory_threshold / (1024**3):.2f} GB")
        logger.info(f"  Cleanup frequency: {cleanup_frequency} steps")
    
    def get_memory_info(self) -> Dict[str, Any]:
        """Get current memory information."""
        memory = psutil.virtual_memory()
        gpu_memory = None
        
        if torch.cuda.is_available():
            gpu_memory = {
                "allocated": torch.cuda.memory_allocated(),
                "reserved": torch.cuda.memory_reserved(),
                "max_allocated": torch.cuda.max_memory_allocated(),
                "max_reserved": torch.cuda.max_memory_reserved()
            }
        
        return {
            "cpu_memory": {
                "total": memory.total,
                "available": memory.available,
                "used": memory.used,
                "percent": memory.percent
            },
            "gpu_memory": gpu_memory,
            "memory_limit": self.memory_limit,
            "memory_threshold": self.memory_threshold,
            "is_over_limit": memory.used > self.memory_threshold
        }
    
    def check_memory_usage(self) -> bool:
        """Check if memory usage is within limits."""
        memory = psutil.virtual_memory()
        return memory.used < self.memory_threshold
    
    def cleanup_memory(self, force: bool = False):
        """
        Clean up memory to prevent overflow.
        
        Args:
            force: Force cleanup even if memory usage is low
        """
        if force or not self.check_memory_usage():
            logger.debug("Cleaning up memory...")
            
            # Python garbage collection
            collected = gc.collect()
            logger.debug(f"Garbage collected {collected} objects")
            
            # CUDA memory cleanup
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                logger.debug("CUDA cache cleared")
            
            # Log memory usage after cleanup
            memory_info = self.get_memory_info()
            logger.debug(f"Memory after cleanup: {memory_info['cpu_memory']['percent']:.1f}%")
    
    def step(self):
        """Increment step counter and perform periodic cleanup."""
        self.step_count += 1
        
        if self.step_count % self.cleanup_frequency == 0:
            self.cleanup_memory()
    
    def log_memory_usage(self, prefix: str = ""):
        """Log current memory usage."""
        memory_info = self.get_memory_info()
        cpu_mem = memory_info["cpu_memory"]
        
        log_msg = f"{prefix}Memory usage: {cpu_mem['percent']:.1f}% "
        log_msg += f"({cpu_mem['used'] / (1024**3):.2f} GB / {cpu_mem['total'] / (1024**3):.2f} GB)"
        
        if memory_info["gpu_memory"]:
            gpu_mem = memory_info["gpu_memory"]
            log_msg += f" | GPU: {gpu_mem['allocated'] / (1024**3):.2f} GB allocated"
        
        logger.info(log_msg)
        
        if memory_info["is_over_limit"]:
            logger.warning(f"{prefix}Memory usage exceeds limit ({self.memory_limit * 100:.1f}%)")
    
    @contextmanager
    def memory_context(self, operation_name: str = "operation"):
        """Context manager for memory monitoring during operations."""
        logger.debug(f"Starting {operation_name}")
        self.log_memory_usage(f"[{operation_name}] ")
        
        try:
            yield self
        finally:
            self.cleanup_memory()
            logger.debug(f"Completed {operation_name}")
            self.log_memory_usage(f"[{operation_name}] ")


class MemoryEfficientTrainer:
    """
    Memory-efficient trainer wrapper with automatic cleanup.
    """
    
    def __init__(self, trainer, memory_monitor: Optional[MemoryMonitor] = None):
        """
        Initialize memory-efficient trainer.
        
        Args:
            trainer: Original trainer object
            memory_monitor: Memory monitor instance
        """
        self.trainer = trainer
        self.memory_monitor = memory_monitor or MemoryMonitor()
        self.step_count = 0
    
    def training_step(self, batch, batch_idx):
        """Memory-efficient training step."""
        # Check memory before training step
        if not self.memory_monitor.check_memory_usage():
            logger.warning("Memory usage high before training step, cleaning up...")
            self.memory_monitor.cleanup_memory(force=True)
        
        # Perform training step
        result = self.trainer.training_step(batch, batch_idx)
        
        # Increment step counter and perform periodic cleanup
        self.memory_monitor.step()
        self.step_count += 1
        
        # Log memory usage periodically
        if self.step_count % 50 == 0:
            self.memory_monitor.log_memory_usage(f"[Step {self.step_count}] ")
        
        return result
    
    def validation_step(self, batch, batch_idx):
        """Memory-efficient validation step."""
        # Check memory before validation step
        if not self.memory_monitor.check_memory_usage():
            logger.warning("Memory usage high before validation step, cleaning up...")
            self.memory_monitor.cleanup_memory(force=True)
        
        # Perform validation step
        result = self.trainer.validation_step(batch, batch_idx)
        
        # Clean up after validation
        self.memory_monitor.cleanup_memory()
        
        return result
    
    def fit(self, data_module):
        """Memory-efficient training."""
        # Delegate to the original trainer
        return self.trainer.fit(data_module)


def optimize_model_for_memory(model: torch.nn.Module) -> torch.nn.Module:
    """
    Optimize model for memory efficiency.
    
    Args:
        model: PyTorch model to optimize
        
    Returns:
        Optimized model
    """
    # Enable gradient checkpointing if available
    if hasattr(model, 'gradient_checkpointing_enable'):
        model.gradient_checkpointing_enable()
        logger.info("Enabled gradient checkpointing")
    
    # Set model to evaluation mode for memory efficiency during inference
    if hasattr(model, 'eval'):
        model.eval()
    
    return model


def create_memory_efficient_dataloader(dataset, batch_size: int = 1, num_workers: int = 2):
    """
    Create memory-efficient dataloader.
    
    Args:
        dataset: PyTorch dataset
        batch_size: Batch size (reduced for memory)
        num_workers: Number of workers (reduced for memory)
        
    Returns:
        Memory-efficient dataloader
    """
    from torch.utils.data import DataLoader
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=False,  # Disabled to save memory
        drop_last=True,
        persistent_workers=False,  # Disabled to save memory
        prefetch_factor=2,  # Reduced prefetch factor
        multiprocessing_context=None  # Use default context
    )


def monitor_system_resources():
    """Monitor system resources and log warnings if needed."""
    # CPU usage
    cpu_percent = psutil.cpu_percent(interval=1)
    if cpu_percent > 90:
        logger.warning(f"High CPU usage: {cpu_percent:.1f}%")
    
    # Memory usage
    memory = psutil.virtual_memory()
    if memory.percent > 90:
        logger.warning(f"High memory usage: {memory.percent:.1f}%")
    
    # Disk usage
    disk = psutil.disk_usage('/')
    if disk.percent > 90:
        logger.warning(f"High disk usage: {disk.percent:.1f}%")
    
    # GPU memory (if available)
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated()
        if gpu_memory > 0.9:
            logger.warning(f"High GPU memory usage: {gpu_memory * 100:.1f}%")


def setup_memory_efficient_training():
    """Setup memory-efficient training environment."""
    # Set PyTorch memory management
    torch.backends.cudnn.benchmark = False  # Disable for reproducibility
    torch.backends.cudnn.deterministic = True
    
    # Set memory fraction if using CUDA
    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(0.8)  # Use max 80% of GPU memory
        logger.info("Set GPU memory fraction to 80%")
    
    # Enable memory efficient attention if available
    try:
        torch.backends.cuda.enable_flash_sdp(True)
        logger.info("Enabled Flash Attention for memory efficiency")
    except:
        logger.debug("Flash Attention not available")
    
    logger.info("Memory-efficient training environment setup complete")


# Global memory monitor instance
_global_memory_monitor = None

def get_global_memory_monitor() -> MemoryMonitor:
    """Get global memory monitor instance."""
    global _global_memory_monitor
    if _global_memory_monitor is None:
        _global_memory_monitor = MemoryMonitor()
    return _global_memory_monitor

def cleanup_global_memory():
    """Clean up global memory."""
    global _global_memory_monitor
    if _global_memory_monitor is not None:
        _global_memory_monitor.cleanup_memory(force=True)
