
from PIL import Image
import sys

def search_grey_parts(image_path):
    try:
        img = Image.open(image_path)
        img = img.convert("RGBA")
        pixels = img.load()
        width, height = img.size
        
        visited = set()
        sprites = []
        
        # Color Classifier for Grey/Metallic which engines likely are
        def is_grey(r, g, b):
            return r > 80 and g > 80 and b > 80 and abs(r-g)<15 and abs(r-b)<15 and abs(g-b)<15

        for y in range(height):
            for x in range(width):
                if (x, y) in visited: continue
                _, _, _, a = pixels[x, y]
                
                if a > 50:
                    q = [(x, y)]
                    visited.add((x, y))
                    min_x, max_x = x, x
                    min_y, max_y = y, y
                    
                    grey_pixels = 0
                    total_pixels = 0
                    
                    idx = 0
                    while idx < len(q):
                        cx, cy = q[idx]
                        idx += 1
                        total_pixels += 1
                        
                        pr, pg, pb, pa = pixels[cx, cy]
                        if is_grey(pr, pg, pb): grey_pixels += 1
                        
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
                    
                    if 20 < w < 200 and 20 < h < 200:
                        if grey_pixels > total_pixels * 0.4:
                            sprites.append({
                                "x": min_x, "y": min_y, "w": w, "h": h,
                                "pixels": total_pixels
                            })

        sprites.sort(key=lambda s: s['y'])
        
        print(f"Found {len(sprites)} GREY candidates:")
        for s in sprites:
            print(f"x={s['x']}, y={s['y']}, w={s['w']}, h={s['h']} (px={s['pixels']})")

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    search_grey_parts("web/frontend/public/spritesheet.png")
