import bpy
import sys
import mathutils

argv = sys.argv
try:
    if "--" in argv:
        argv = argv[argv.index("--") + 1:]
    else:
        # If no -- present, just take the last two args if running via blender python directly
        pass

    file1 = argv[0]
    
    print(f"Checking: {file1}")
    
    bpy.ops.wm.read_factory_settings(use_empty=True)
    
    if file1.endswith(".obj"):
        if hasattr(bpy.ops.wm, "obj_import"):
             bpy.ops.wm.obj_import(filepath=file1)
        else:
             bpy.ops.import_scene.obj(filepath=file1)
    elif file1.endswith(".glb"):
        bpy.ops.import_scene.gltf(filepath=file1)
        
    # Get dimensions
    min_vec = mathutils.Vector((float('inf'), float('inf'), float('inf')))
    max_vec = mathutils.Vector((float('-inf'), float('-inf'), float('-inf')))
    
    found = False
    for obj in bpy.context.selected_objects:
        if obj.type == 'MESH':
            found = True
            # Apply transform to dimensions?
            # dimensions is local but applied with scale
            # let's calculate global bounds
            mw = obj.matrix_world
            if obj.bound_box:
                for corner in obj.bound_box:
                    global_corner = mw @ mathutils.Vector(corner)
                    min_vec.x = min(min_vec.x, global_corner.x)
                    min_vec.y = min(min_vec.y, global_corner.y)
                    min_vec.z = min(min_vec.z, global_corner.z)
                    max_vec.x = max(max_vec.x, global_corner.x)
                    max_vec.y = max(max_vec.y, global_corner.y)
                    max_vec.z = max(max_vec.z, global_corner.z)

    if found:
        size = max_vec - min_vec
        print(f"Dimensions: {size}")
        print(f"Max Z: {max_vec.z}")
    else:
        print("No mesh found")

except Exception as e:
    print(f"Error: {e}")
