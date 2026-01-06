import math

# Constants
WIDTH = 16000
HEIGHT = 9000
POD_RADIUS = 400.0
CHECKPOINT_RADIUS = 600.0
MAX_TURN_DEGREES = 18.0
FRICTION = 0.85

def dist(p1_x, p1_y, p2_x, p2_y):
    return math.sqrt((p1_x - p2_x)**2 + (p1_y - p2_y)**2)

def dist_sq(p1_x, p1_y, p2_x, p2_y):
    return (p1_x - p2_x)**2 + (p1_y - p2_y)**2

def clip(val, min_val, max_val):
    return max(min_val, min(max_val, val))

def sim_step(pods, cps, actions):
    """
    pods: list of dicts {'x','y','vx','vy','a','s','n','to', 'laps'}
            s=shield, n=next_cp_id, to=timeout
    cps: list of [x, y]
    actions: list of [thrust, angle, shield_active] (for each pod 0..3)
    """
    
    # 1. Rotate & Thrust
    for i, p in enumerate(pods):
        ua = actions[i][1] # Desired Angle Change (Relative) OR Absolute Target?
        # Standard input is Target X, Y, Thrust.
        # But here we are searching with 'Angle' variations relative to current.
        # Let's assume action[1] is DELTA angle (-18..18)
        
        # Rotate
        da = clip(ua, -MAX_TURN_DEGREES, MAX_TURN_DEGREES)
        p['a'] += da
        # Normalize
        if p['a'] > 180: p['a'] -= 360
        elif p['a'] <= -180: p['a'] += 360
        
        # Thrust
        th = clip(actions[i][0], 0, 100)
        
        # Shield
        shield_active = actions[i][2]
        
        if shield_active:
             p['s'] = 4 # Activation duration
             p['m'] = 10.0
             th = 0 # Cannot thrust while shielding? Rule check needed.
             # "Prevents accelerating for the next 3 turns"
             # If shielding THIS turn, we don't accelerate.
        else:
             if p['s'] > 0: p['s'] -= 1
             p['m'] = 1.0 if p['s'] == 0 else 10.0 # Mass legacy? 
             # Simpler: Shield=4 -> Mass=10. Shield=3,2,1 -> Mass=1.0 but no thrust?
             # Rule: "Increases mass x10 for 1 turn. Prevents accelerating for 3 turns."
             # So if p['s'] > 0, th = 0.
             # Only if we JUST activated (s=4) is mass 10.
             if p['s'] == 3: p['m'] = 10.0 # Wait, countdown goes 3->2->1->0?
             else: p['m'] = 1.0
             
             if p['s'] > 0: th = 0 
        
        rad = math.radians(p['a'])
        p['vx'] += th * math.cos(rad)
        p['vy'] += th * math.sin(rad)
        
        # Move
        p['x'] += p['vx']
        p['y'] += p['vy']

    # 2. Collisions
    # Simple Elastic Iterative (K=2 is usually enough for local search)
    for _ in range(2): 
        imp = [[0.0]*4 for _ in range(4)] # x, y, sx, sy (impulse + separation)
        
        pairs = []
        for i in range(4):
            for j in range(i+1, 4):
                 dx = pods[j]['x'] - pods[i]['x']
                 dy = pods[j]['y'] - pods[i]['y']
                 d2 = dx*dx + dy*dy
                 r_sum = POD_RADIUS * 2
                 if d2 < r_sum*r_sum:
                     d = math.sqrt(d2)
                     if d < 1e-4: d=1e-4
                     nx, ny = dx/d, dy/d
                     
                     dvx = pods[i]['vx'] - pods[j]['vx']
                     dvy = pods[i]['vy'] - pods[j]['vy']
                     
                     prod = dvx*nx + dvy*ny
                     m1 = pods[i].get('m', 1.0)
                     m2 = pods[j].get('m', 1.0)
                     
                     f = prod / (1/m1 + 1/m2)
                     
                     # Min Impulse 120
                     # ONLY if impact is strong enough? No, min impulse always applies on impact?
                     # Rules: "The minimum impulse is 120."
                     if f < 120.0 and prod > 0: # Moving towards each other?
                         # If prod > 0 -> v1 faster towards v2 (collision).
                         # If prod < 0 -> separating (no impulse needed usually, but overlap exists)
                         f = 120.0
                     elif prod <= 0:
                         f = 0
                     
                     # J = -F * N
                     jx, jy = -f*nx, -f*ny
                     
                     # Sep
                     overlap = r_sum - d
                     sep = overlap / 2.0
                     sx, sy = nx*sep, ny*sep
                     
                     pairs.append((i, j, jx, jy, sx, sy, m1, m2))
        
        # Apply
        for i, j, jx, jy, sx, sy, m1, m2 in pairs:
             pods[i]['vx'] += jx / m1
             pods[i]['vy'] += jy / m1
             pods[i]['x'] -= sx
             pods[i]['y'] -= sy
             
             pods[j]['vx'] -= jx / m2
             pods[j]['vy'] -= jy / m2
             pods[j]['x'] += sx
             pods[j]['y'] += sy
             
             # Friction Bounce Factor? No mentioned in basic rules, just 120 imp.

    # 3. Checkpoints & Friction
    for p in pods:
        p['vx'] *= FRICTION
        p['vy'] *= FRICTION
        p['x'] = round(p['x'])
        p['y'] = round(p['y'])
        p['vx'] = int(p['vx'])
        p['vy'] = int(p['vy'])
        
        # CP Check
        cp = cps[p['n']]
        d2 = dist_sq(p['x'], p['y'], cp[0], cp[1])
        if d2 < CHECKPOINT_RADIUS**2:
            p['n'] = (p['n'] + 1)
            if p['n'] >= len(cps):
                p['n'] = 0
                p['laps'] += 1
            p['to'] = 100
        else:
            p['to'] -= 1
            
    return pods
