import os
import glob
from PIL import Image

def compress_and_replace():
    # Define directories
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    assets_dir = os.path.join(base_dir, "docs", "assets")
    docs_dir = os.path.join(base_dir, "docs")
    readme_path = os.path.join(base_dir, "README.md")
    readme_zh_path = os.path.join(base_dir, "README_ZH.md")

    # Find all PNG files
    png_files = glob.glob(os.path.join(assets_dir, "*.png"))
    
    if not png_files:
        print("No PNG files found in docs/assets.")
        return

    print(f"Found {len(png_files)} PNG files. Starting conversion...")

    total_saved = 0
    replacement_map = {}

    for png_path in png_files:
        filename = os.path.basename(png_path)
        jpg_filename = filename.replace(".png", ".jpg")
        jpg_path = os.path.join(assets_dir, jpg_filename)

        # Convert to JPG
        try:
            with Image.open(png_path) as img:
                rgb_im = img.convert('RGB')
                rgb_im.save(jpg_path, quality=80, optimize=True)
            
            # Calculate stats
            old_size = os.path.getsize(png_path)
            new_size = os.path.getsize(jpg_path)
            saved = old_size - new_size
            total_saved += saved
            
            print(f"Converted: {filename} ({old_size/1024:.1f}KB) -> {jpg_filename} ({new_size/1024:.1f}KB). Saved {saved/1024:.1f}KB")
            
            # Add to map for link replacement
            replacement_map[filename] = jpg_filename
            
            # Remove original PNG
            os.remove(png_path)
            
        except Exception as e:
            print(f"Error converting {filename}: {e}")

    print(f"Total space saved: {total_saved/1024/1024:.2f} MB")
    print("Updating Markdown links...")

    # Find all Markdown files to update
    md_files = glob.glob(os.path.join(docs_dir, "**", "*.md"), recursive=True)
    md_files.append(readme_path)
    md_files.append(readme_zh_path)

    for md_file in md_files:
        if not os.path.exists(md_file):
            continue
            
        with open(md_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        new_content = content
        count = 0
        for png_name, jpg_name in replacement_map.items():
            if png_name in new_content:
                new_content = new_content.replace(png_name, jpg_name)
                count += 1
        
        if count > 0:
            with open(md_file, 'w', encoding='utf-8') as f:
                f.write(new_content)
            print(f"Updated {count} links in {os.path.basename(md_file)}")

    print("Done! All PNGs converted to JPG and links updated.")

if __name__ == "__main__":
    compress_and_replace()
