import os
import subprocess
import sys
try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False


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

# Create Material and Link Textures
def setup_material(obj_object, input_path):
    # Ensure usage of nodes
    if not obj_object.data.materials:
        mat = bpy.data.materials.new(name="Material")
        obj_object.data.materials.append(mat)
    else:
        mat = obj_object.data.materials[0]
    
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    
    # Clear default nodes
    nodes.clear()
    
    # Create Principled BSDF and Output
    node_princ = nodes.new(type='ShaderNodeBsdfPrincipled')
    node_princ.location = (0, 0)
    node_out = nodes.new(type='ShaderNodeOutputMaterial')
    node_out.location = (300, 0)
    links.new(node_princ.outputs['BSDF'], node_out.inputs['Surface'])
    
    # Texture Paths Derivation
    # Rule: 
    #   Normal Map: [filename].png (same name as OBJ) -> e.g. StoneQuarry_SPT_mesh.png
    #   Base Color: [filename_without_mesh].png -> e.g. StoneQuarry_SPT.png
    
    base_dir = os.path.dirname(input_path)
    filename = os.path.basename(input_path)
    name_no_ext = os.path.splitext(filename)[0]
    
    # Normal Path: matches OBJ name
    normal_path = os.path.join(base_dir, name_no_ext + ".png")
    
    # Base Path: remove "_mesh" if present
    base_name_clean = name_no_ext.replace("_mesh", "")
    base_path = os.path.join(base_dir, base_name_clean + ".png")
    
    print(f"Looking for textures:")
    print(f"  Normal: {normal_path}")
    print(f"  Base:   {base_path}")
    
    # Helper to load image
    def load_image_node(path, is_data=False):
        if os.path.exists(path):
            try:
                img = bpy.data.images.load(path)
                node_tex = nodes.new(type='ShaderNodeTexImage')
                node_tex.image = img
                if is_data:
                    img.colorspace_settings.name = 'Non-Color' # Updated syntax for non-color data
                return node_tex
            except Exception as e:
                print(f"Failed to load image {path}: {e}")
        return None

    # Load Base Color
    node_base = load_image_node(base_path, is_data=False)
    if node_base:
        node_base.location = (-400, 200)
        links.new(node_base.outputs['Color'], node_princ.inputs['Base Color'])
        print(f"  -> Linked Base Color: {base_path}")
        
    # Load Normal Map
    node_normal = load_image_node(normal_path, is_data=True)
    if node_normal:
        node_normal.location = (-600, -200)
        
        # Create Normal Map Node
        node_normal_map = nodes.new(type='ShaderNodeNormalMap')
        node_normal_map.location = (-300, -200)
        
        links.new(node_normal.outputs['Color'], node_normal_map.inputs['Color'])
        links.new(node_normal_map.outputs['Normal'], node_princ.inputs['Normal'])
        print(f"  -> Linked Normal Map: {normal_path}")

# Run import
# Clear existing mesh objects
bpy.ops.wm.read_factory_settings(use_empty=True)

# Import OBJ
if hasattr(bpy.ops.wm, "obj_import"):
    bpy.ops.wm.obj_import(filepath=input_file)
else:
    bpy.ops.import_scene.obj(filepath=input_file, use_split_objects=False, use_split_groups=False, global_clamp_size=0.0)

# Process imported objects
for obj in bpy.context.selected_objects:
    if obj.type == 'MESH':
        setup_material(obj, input_file)

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
    if not PIL_AVAILABLE:
        print(f"Skipping resize for {png_path} (PIL not installed)")
        return

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
