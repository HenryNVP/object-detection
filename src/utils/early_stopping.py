"""Early stopping utility for training."""

from __future__ import annotations


class EarlyStopping:
    """Early stopping to stop training when validation loss doesn't improve.
    
    Args:
        patience: Number of epochs to wait for improvement before stopping
        min_delta: Minimum change in monitored value to qualify as improvement
        mode: One of 'min' or 'max'. In 'min' mode, training stops when metric stops decreasing.
        verbose: If True, prints messages when patience counter increases
    """
    
    def __init__(
        self,
        patience: int = 2,
        min_delta: float = 0.0,
        mode: str = "min",
        verbose: bool = True,
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.verbose = verbose
        
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_epoch = 0
        
        if mode == "min":
            self.monitor_op = lambda current, best: current < best - min_delta
        elif mode == "max":
            self.monitor_op = lambda current, best: current > best + min_delta
        else:
            raise ValueError(f"mode must be 'min' or 'max', got {mode}")
    
    def __call__(self, current_score: float, epoch: int) -> bool:
        """Check if training should stop.
        
        Args:
            current_score: Current metric value
            epoch: Current epoch number
            
        Returns:
            True if training should stop, False otherwise
        """
        if self.best_score is None:
            self.best_score = current_score
            self.best_epoch = epoch
            return False
        
        if self.monitor_op(current_score, self.best_score):
            # Improvement
            self.best_score = current_score
            self.best_epoch = epoch
            self.counter = 0
            if self.verbose:
                print(f"  ✓ Validation improved (patience reset: {self.counter}/{self.patience})")
        else:
            # No improvement
            self.counter += 1
            if self.verbose:
                print(f"  ⚠️  No improvement for {self.counter} epoch(s) (patience: {self.counter}/{self.patience})")
            
            if self.counter >= self.patience:
                self.early_stop = True
                if self.verbose:
                    print(f"\n🛑 Early stopping triggered!")
                    print(f"   Best epoch: {self.best_epoch}")
                    print(f"   Best val loss: {self.best_score:.4f}")
                return True
        
        return False
    
    def reset(self):
        """Reset the early stopping state."""
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_epoch = 0

