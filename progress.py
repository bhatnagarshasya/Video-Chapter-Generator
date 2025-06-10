from tqdm import tqdm
from typing import Optional, Callable
import time

class ProgressTracker:
    def __init__(self, total_steps: int, description: str = "Processing"):
        self.total_steps = total_steps
        self.current_step = 0
        self.description = description
        self.start_time = time.time()
        self.pbar = tqdm(total=total_steps, desc=description)
        
    def update(self, step: int = 1, description: Optional[str] = None):
        """Update progress bar."""
        self.current_step += step
        if description:
            self.pbar.set_description(description)
        self.pbar.update(step)
        
    def set_description(self, description: str):
        """Set progress bar description."""
        self.pbar.set_description(description)
        
    def close(self):
        """Close progress bar."""
        self.pbar.close()
        
    def get_elapsed_time(self) -> float:
        """Get elapsed time in seconds."""
        return time.time() - self.start_time

def track_progress(total_steps: int, description: str = "Processing") -> Callable:
    """Decorator for tracking progress of a function."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            tracker = ProgressTracker(total_steps, description)
            try:
                result = func(*args, **kwargs, progress_tracker=tracker)
                return result
            finally:
                tracker.close()
        return wrapper
    return decorator 