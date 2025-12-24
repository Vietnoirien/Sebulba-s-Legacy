
from PIL import Image
import sys

def search_columns(image_path):
    try:
        img = Image.open(image_path)
        img = img.convert("RGBA")
        pixels = img.load()
        
        # Search Column X=950 to 1050 (Engines?)
        # Search Column X=1100 to 1450 (Hulls/Cabins?)
        
        columns = [
            (950, 1050, "Engines Column"),
            (1100, 1450, "Cabins Column")
        ]
        
        filtered_sprites = []
        
        visited = set()
        
        for min_x, max_x, label in columns:
            print(f"Scanning {label} x:{min_x}-{max_x}...")
            for y in range(400, img.height):
                for x in range(min_x, max_x):
                    if (x, y) in visited: continue
                    _, _, _, a = pixels[x, y]
                    
                    if a > 40:
                        q = [(x, y)]
                        visited.add((x, y))
                        min_x_s, max_x_s = x, x
                        min_y_s, max_y_s = y, y
                        
                        idx = 0
                        while idx < len(q):
                            cx, cy = q[idx]
                            idx += 1
                            min_x_s = min(min_x_s, cx); max_x_s = max(max_x_s, cx)
                            min_y_s = min(min_y_s, cy); max_y_s = max(max_y_s, cy)
                            
                            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                                nx, ny = cx + dx, cy + dy
                                if min_x <= nx < max_x and 400 <= ny < img.height:
                                    if (nx, ny) not in visited:
                                        _, _, _, na = pixels[nx, ny]
                                        if na > 40:
                                            visited.add((nx, ny))
                                            q.append((nx, ny))
                                            
                        w = max_x_s - min_x_s + 1
                        h = max_y_s - min_y_s + 1
                        
                        if w > 20 and h > 40: # Filter small noise
                            filtered_sprites.append({
                                "x": min_x_s, "y": min_y_s, "w": w, "h": h,
                                "column": label
                            })

        filtered_sprites.sort(key=lambda s: (s['column'], s['y']))
        
        for s in filtered_sprites:
             print(f"Candidate [{s['column']}]: x={s['x']}, y={s['y']}, w={s['w']}, h={s['h']}")

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    search_columns("web/frontend/public/spritesheet.png")
