import torch
import time

def allocate_large_tensor(gb_to_allocate=10):
    """
    Allocates a large tensor on GPU for testing memory usage.

    Args:
        gb_to_allocate (int): The number of gigabytes to allocate.
    """
    if not torch.cuda.is_available():
        raise EnvironmentError("CUDA is not available on this machine.")

    device = torch.device("cuda")
    total_bytes = gb_to_allocate * 1024**3
    float_tensor_size = torch.tensor([], dtype=torch.float32).element_size()  # Typically 4 bytes
    num_elements = total_bytes // float_tensor_size

    print(f"Allocating {gb_to_allocate} GB ({num_elements} float32 elements) on {device}...")
    try:
        tensor = torch.empty(num_elements, dtype=torch.float32, device=device)
        tensor.fill_(1.0)  # Optional: fill to ensure memory is actually used
        print("Allocation successful. Tensor is on GPU.")
        print("Press Enter to release the tensor and exit...")
        input()  # Wait for user input before releasing the tensor
    except RuntimeError as e:
        print(f"Allocation failed: {e}")

if __name__ == "__main__":
    allocate_large_tensor(gb_to_allocate=10)  # Change value for different loads