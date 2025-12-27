
from PIL import Image
import sys

def find_hulls(image_path):
    try:
        img = Image.open(image_path)
        img = img.convert("RGBA")
        pixels = img.load()
        width, height = img.size
        
        visited = set()
        sprites = []
        
        # Look at the bottom half where ships are likely to be
        for y in range(400, height):
            for x in range(0, width): # Check full width of bottom
                if (x, y) in visited:
                    continue
                
                _, _, _, a = pixels[x, y]
                if a > 40: # Reasonable threshold
                    q = [(x, y)]
                    visited.add((x, y))
                    min_x, max_x = x, x
                    min_y, max_y = y, y
                    
                    count = 0
                    sum_r, sum_g, sum_b = 0, 0, 0
                    
                    idx = 0
                    while idx < len(q):
                        cx, cy = q[idx]
                        idx += 1
                        count += 1
                        
                        pr, pg, pb, pa = pixels[cx, cy]
                        sum_r += pr
                        sum_g += pg
                        sum_b += pb
                        
                        min_x = min(min_x, cx)
                        max_x = max(max_x, cx)
                        min_y = min(min_y, cy)
                        max_y = max(max_y, cy)
                        
                        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                            nx, ny = cx + dx, cy + dy
                            if 0 <= nx < width and 400 <= ny < height:
                                if (nx, ny) not in visited:
                                    nr, ng, nb, na = pixels[nx, ny]
                                    if na > 40:
                                        visited.add((nx, ny))
                                        q.append((nx, ny))
                    
                    w = max_x - min_x + 1
                    h = max_y - min_y + 1
                    
                    # Hull size filter: typically around 80-120px wide/high
                    if w > 40 and h > 40:
                        sprites.append({
                            "x": min_x, 
                            "y": min_y, 
                            "w": w, 
                            "h": h,
                            "rgb": (sum_r//count, sum_g//count, sum_b//count)
                        })

        sprites.sort(key=lambda s: s["x"])
        
        print(f"Found {len(sprites)} potential hull parts:")
        for s in sprites:
             print(f"Hull Candidate: x={s['x']}, y={s['y']}, w={s['w']}, h={s['h']}, RGB={s['rgb']}")

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    find_hulls("web/frontend/public/spritesheet.png")
