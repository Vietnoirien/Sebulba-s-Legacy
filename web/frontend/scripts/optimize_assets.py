import os
import subprocess
import sys
from PIL import Image

# Configuration
ASSETS_DIR = os.path.abspath("web/frontend/src/assets/models")
BLENDER_BIN = "blender" # Assumes blender is in PATH
MAX_TEXTURE_SIZE = 1024

# Blender conversion script
BLENDER_SCRIPT = """
import bpy
import sys
import os

argv = sys.argv
argv = argv[argv.index("--") + 1:] # get all args after "--"

input_file = argv[0]
output_file = argv[1]

# Clear existing mesh objects
bpy.ops.wm.read_factory_settings(use_empty=True)

# Import OBJ
# Attempt to use the new importer (Blender 4.0+)
if hasattr(bpy.ops.wm, "obj_import"):
    # Ensure no clamping, and import as single object if possible to keep relative offsets
    # New importer defaults to no clamping usually, or use clamp_size=0.0 if needed, but let's try defaults.
    bpy.ops.wm.obj_import(filepath=input_file)
else:
    # Fallback for older Blender versions
    # split_mode='OFF' helps keep the object as a single mesh with original origin
    bpy.ops.import_scene.obj(filepath=input_file, use_split_objects=False, use_split_groups=False, global_clamp_size=0.0)

# Export GLB
bpy.ops.export_scene.gltf(filepath=output_file, export_format='GLB')
"""

def create_blender_script():
    with open("temp_convert.py", "w") as f:
        f.write(BLENDER_SCRIPT)

def convert_obj_to_glb(obj_path):
    glb_path = obj_path.replace(".obj", ".glb")
    
    if os.path.exists(glb_path):
        print(f"Skipping {os.path.basename(obj_path)} (GLB exists)")
        return

    print(f"Converting {os.path.basename(obj_path)} to GLB...")
    
    cmd = [
        BLENDER_BIN,
        "--background",
        "--python", "temp_convert.py",
        "--",
        obj_path,
        glb_path
    ]
    
    print(f"  Target: {glb_path}")
    
    try:
        # Remove stdout/stderr suppression to see Blender errors
        subprocess.run(cmd, check=True) 
        if os.path.exists(glb_path):
            print(f"  -> Created {os.path.basename(glb_path)}")
        else:
            print(f"  -> FAILURE: Blender exited successfully but {glb_path} does not exist!")
    except subprocess.CalledProcessError as e:
        print(f"  -> Error converting {obj_path}: {e}")

def resize_texture(png_path):
    try:
        with Image.open(png_path) as img:
            if img.width > MAX_TEXTURE_SIZE or img.height > MAX_TEXTURE_SIZE:
                print(f"Resizing {os.path.basename(png_path)} ({img.width}x{img.height})...")
                img.thumbnail((MAX_TEXTURE_SIZE, MAX_TEXTURE_SIZE))
                img.save(png_path)
                print(f"  -> Resized to {img.width}x{img.height}")
    except Exception as e:
        print(f"Error resizing {png_path}: {e}")

def main():
    print(f"Scanning {ASSETS_DIR}...")
    create_blender_script()

    for root, dirs, files in os.walk(ASSETS_DIR):
        for file in files:
            if file.endswith(".obj"):
                obj_path = os.path.join(root, file)
                convert_obj_to_glb(obj_path)
            elif file.endswith(".png"):
                 # Check if it's a skin texture (usually large)
                 if "skin" in file.lower() or "checkpoint" in file.lower():
                     png_path = os.path.join(root, file)
                     resize_texture(png_path)

    if os.path.exists("temp_convert.py"):
        os.remove("temp_convert.py")
    
    print("Optimization complete.")

if __name__ == "__main__":
    main()
