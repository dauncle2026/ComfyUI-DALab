import os
import glob
from PIL import Image

def compress_jpgs():
    # Define directories
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    workflows_dir = os.path.join(base_dir, "example_workflows")

    # Find all JPG files
    jpg_files = glob.glob(os.path.join(workflows_dir, "*.jpg"))
    
    if not jpg_files:
        print("No JPG files found in example_workflows.")
        return

    print(f"Found {len(jpg_files)} JPG files. Starting compression...")

    total_saved = 0
    count = 0

    for jpg_path in jpg_files:
        filename = os.path.basename(jpg_path)

        try:
            old_size = os.path.getsize(jpg_path)
            
            # Compress
            with Image.open(jpg_path) as img:
                # Save to temp file
                temp_path = jpg_path + ".temp"
                img.save(temp_path, "JPEG", quality=80, optimize=True)
            
            new_size = os.path.getsize(temp_path)
            
            if new_size < old_size:
                saved = old_size - new_size
                total_saved += saved
                count += 1
                
                # Replace original
                os.remove(jpg_path)
                os.rename(temp_path, jpg_path)
                
                print(f"Compressed: {filename} ({old_size/1024:.1f}KB -> {new_size/1024:.1f}KB). Saved {saved/1024:.1f}KB")
            else:
                print(f"Skipped: {filename} (No saving: {old_size/1024:.1f}KB -> {new_size/1024:.1f}KB)")
                os.remove(temp_path)
                
        except Exception as e:
            print(f"Error compressing {filename}: {e}")
            if os.path.exists(jpg_path + ".temp"):
                 os.remove(jpg_path + ".temp")

    print(f"Compressed {count} files.")
    print(f"Total space saved: {total_saved/1024/1024:.2f} MB")

if __name__ == "__main__":
    compress_jpgs()
