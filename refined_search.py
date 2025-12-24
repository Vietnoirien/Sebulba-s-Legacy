
from PIL import Image
import sys

def refined_search(image_path):
    try:
        img = Image.open(image_path)
        img = img.convert("RGBA")
        pixels = img.load()
        width, height = img.size
        
        print(f"Image Size: {width}x{height}")
        
        visited = set()
        sprites = []
        
        # Color Classifiers
        def get_color_type(r, g, b):
            # Red/Brown
            if r > 100 and g < 90 and b < 90: return "RED"
            # White/Grey
            if r > 180 and g > 180 and b > 180: return "WHITE"
            # Metallic Grey (Engines often are)
            if r > 100 and g > 100 and b > 100 and abs(r-g)<20 and abs(r-b)<20: return "GREY"
            return None

        # Ignore known UI areas if possible, but let's just filter by size
        # Engines/Cabins are likely between 20x40 and 100x150
        
        for y in range(height):
            for x in range(width):
                if (x, y) in visited: continue
                _, _, _, a = pixels[x, y]
                
                if a > 50:
                    q = [(x, y)]
                    visited.add((x, y))
                    min_x, max_x = x, x
                    min_y, max_y = y, y
                    
                    color_counts = {"RED": 0, "WHITE": 0, "GREY": 0}
                    total_pixels = 0
                    
                    idx = 0
                    while idx < len(q):
                        cx, cy = q[idx]
                        idx += 1
                        total_pixels += 1
                        
                        pr, pg, pb, pa = pixels[cx, cy]
                        c_type = get_color_type(pr, pg, pb)
                        if c_type: color_counts[c_type] += 1
                        
                        min_x = min(min_x, cx); max_x = max(max_x, cx)
                        min_y = min(min_y, cy); max_y = max(max_y, cy)
                        
                        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                            nx, ny = cx + dx, cy + dy
                            if 0 <= nx < width and 0 <= ny < height:
                                if (nx, ny) not in visited:
                                    _, _, _, na = pixels[nx, ny]
                                    if na > 50:
                                        visited.add((nx, ny))
                                        q.append((nx, ny))
                    
                    w = max_x - min_x + 1
                    h = max_y - min_y + 1
                    
                    # Size Filter for Ship Parts
                    if 20 < w < 200 and 20 < h < 200:
                        # Determine dominant color
                        dom_color = "OTHER"
                        if color_counts["RED"] > total_pixels * 0.3: dom_color = "RED"
                        elif color_counts["WHITE"] > total_pixels * 0.3: dom_color = "WHITE"
                        elif color_counts["GREY"] > total_pixels * 0.3: dom_color = "GREY"
                        
                        if dom_color in ["RED", "WHITE", "GREY"]:
                            sprites.append({
                                "x": min_x, "y": min_y, "w": w, "h": h,
                                "color": dom_color,
                                "pixels": total_pixels
                            })

        # Sort by Color then Y
        sprites.sort(key=lambda s: (s['color'], s['y']))
        
        print(f"Found {len(sprites)} candidates:")
        for s in sprites:
            print(f"[{s['color']}] x={s['x']}, y={s['y']}, w={s['w']}, h={s['h']} (px={s['pixels']})")

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    refined_search("web/frontend/public/spritesheet.png")
