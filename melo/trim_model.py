import os
import torch
import argparse
from pathlib import Path

def clean_checkpoint(input_path, output_path=None):
    """
    Convert a full MeloTTS training checkpoint to an inference-only checkpoint.
    
    Args:
        input_path (str): Path to the full checkpoint
        output_path (str, optional): Path for the clean checkpoint
    """
    print(f"Loading checkpoint: {input_path}")
    
    # Load the checkpoint
    checkpoint = torch.load(input_path, map_location='cpu')
    
    print("\nCheckpoint contents:")
    print("Keys in checkpoint:", list(checkpoint.keys()))
    
    # Create clean checkpoint keeping only necessary components
    clean_checkpoint = {
        'model': checkpoint['model'],  # Keep the model state
        'iteration': checkpoint['iteration'],  # Keep track of training iteration
        'learning_rate': checkpoint['learning_rate'],  # Keep the learning rate
        # Exclude the optimizer state
    }
    
    # Generate output path if not provided
    if output_path is None:
        input_path = Path(input_path)
        output_path = input_path.parent / f"{input_path.stem}_clean{input_path.suffix}"
    
    # Save the clean checkpoint
    print(f"\nSaving clean checkpoint to: {output_path}")
    torch.save(clean_checkpoint, output_path)
    
    # Print size comparison
    original_size = os.path.getsize(input_path) / (1024 * 1024)  # MB
    clean_size = os.path.getsize(output_path) / (1024 * 1024)    # MB
    
    print(f"\nSize comparison:")
    print(f"Original checkpoint: {original_size:.2f} MB")
    print(f"Clean checkpoint: {clean_size:.2f} MB")
    print(f"Size reduction: {((original_size - clean_size) / original_size * 100):.1f}%")
    
    # Verify the clean checkpoint was saved properly
    if os.path.getsize(output_path) == 0:
        raise Exception("Error: Output file is empty!")
    
    # Load the clean checkpoint to verify it's valid
    try:
        test_load = torch.load(output_path, map_location='cpu')
        print("\nVerification successful - cleaned checkpoint contains:")
        print("Keys in cleaned checkpoint:", list(test_load.keys()))
    except Exception as e:
        raise Exception(f"Error verifying cleaned checkpoint: {str(e)}")

def main():
    parser = argparse.ArgumentParser(description='Convert MeloTTS training checkpoint to inference-only checkpoint')
    parser.add_argument('input_path', type=str, help='Path to the full checkpoint file')
    parser.add_argument('--output_path', type=str, default=None, 
                       help='Path for the clean checkpoint (optional)')
    parser.add_argument('--debug', action='store_true', 
                       help='Print additional debugging information')
    
    args = parser.parse_args()
    
    try:
        clean_checkpoint(args.input_path, args.output_path)
        print("\nCheckpoint cleaned successfully!")
    except Exception as e:
        print(f"\nError occurred: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    main()