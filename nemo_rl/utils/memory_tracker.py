# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import gc
from typing import Optional, Dict, Any

class MemoryTracker:
    """
    A simple memory tracker to log CUDA memory usage at different stages of execution.
    """
    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self.reset()

    def reset(self):
        """Reset the memory tracker state."""
        self.stages = []
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

    def snapshot_start_of_stage(self, stage_name: str, local_vars: Optional[Dict[str, Any]] = None):
        """
        Snapshot memory usage at the start of a stage.
        
        Args:
            stage_name: Name of the stage
            local_vars: Optional dictionary of local variables (unused for now, but kept for signature compatibility)
        """
        if not self.enabled or not torch.cuda.is_available():
            return

        # Force garbage collection to get accurate reading
        gc.collect()
        torch.cuda.empty_cache()
        
        allocated = torch.cuda.memory_allocated() / (1024 ** 3)  # GB
        reserved = torch.cuda.memory_reserved() / (1024 ** 3)    # GB
        
        print(f"[MemoryTracker] Stage '{stage_name}': Allocated {allocated:.2f} GB, Reserved {reserved:.2f} GB", flush=True)
        
        self.stages.append({
            "stage": stage_name,
            "allocated_gb": allocated,
            "reserved_gb": reserved
        })
