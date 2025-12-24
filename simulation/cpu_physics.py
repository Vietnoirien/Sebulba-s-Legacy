import math
from config import *

class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def distance(self, p):
        return math.sqrt((self.x - p.x)**2 + (self.y - p.y)**2)

    def distance2(self, p):
        return (self.x - p.x)**2 + (self.y - p.y)**2

class Unit(Point):
    def __init__(self, x, y, vx=0, vy=0, mass=1.0, radius=0.0):
        super().__init__(x, y)
        self.vx = vx
        self.vy = vy
        self.mass = mass
        self.radius = radius

class Checkpoint(Point):
    def __init__(self, x, y, id):
        super().__init__(x, y)
        self.id = id
        self.radius = CHECKPOINT_RADIUS

class Pod(Unit):
    def __init__(self, id, team_id, x, y, angle=0.0):
        super().__init__(x, y, vx=0, vy=0, mass=1.0, radius=POD_RADIUS)
        self.id = id
        self.team_id = team_id
        self.angle = angle # In degrees
        self.next_checkpoint_id = 1
        self.laps = 0
        self.shield = 0 # Shield cooldwon
        self.boost_available = True
        self.timeout = 100
        
        # Action buffer
        self.next_angle = 0.0
        self.next_thrust = 0

    def apply_movement(self, checkpoints):
        # 1. Rotate to desired angle (clamped +- 18 degrees relative to current)
        # Note: The input to this function should already be the verified/clamped angle,
        # but we enforce the constraint here just in case, relative to self.angle.
        # However, usually the "Action" determines the target angle or offset. 
        # The plan says: "Rotate to desired angle (clamped +- 18 deg)"
        # Let's assume self.next_angle is the TARGET angle the pod wants to face.
        
        diff = self.next_angle - self.angle
        # Normalize diff to [-180, 180]
        while diff <= -180: diff += 360
        while diff > 180: diff -= 360
        
        # Clamp to MAX_TURN_DEGREES
        if diff > MAX_TURN_DEGREES: diff = MAX_TURN_DEGREES
        if diff < -MAX_TURN_DEGREES: diff = -MAX_TURN_DEGREES
        
        self.angle += diff
        # Normalize angle
        while self.angle <= -180: self.angle += 360
        while self.angle > 180: self.angle -= 360

        # 2. Add Thrust to Velocity
        angle_rad = math.radians(self.angle)
        
        # Shield logic: If shield is active (just activated), we don't accelerate? 
        # Plan says: "Prevents accelerating for the next 3 turns."
        # If self.shield == 4 (just activated) or > 0?
        # Actually checking Rule 1.3: "Increases mass x10 for 1 turn... Prevents accelerating for the next 3 turns."
        # So if shield count is > 0, we might block thrust?
        # Let's clarify shield state. Usually shield=3 means active for this turn, then cooldown.
        # But standard CodinGame rule: Shield is active when you command it. It lasts 1 turn?
        # "SHIELD: Increases mass x10 for 1 turn (Momentum conservation variant). Prevents accelerating for the next 3 turns."
        # This implies a cooldown mechanism.
        
        # Let's assume strictly following the plan:
        # If we commanded SHIELD this turn, mass is 10.
        # If we have cooldown, we can't thrust.
        
        # However, for simplicity here, let's process thrust normally unless inhibited.
        
        curr_thrust = self.next_thrust
        
        self.vx += curr_thrust * math.cos(angle_rad)
        self.vy += curr_thrust * math.sin(angle_rad)

        # 3. Move Position
        self.x += self.vx
        self.y += self.vy

    def end_frame(self):
        # 5. Apply Friction
        self.vx *= FRICTION
        self.vy *= FRICTION
        
        # Truncate
        self.vx = int(self.vx)
        self.vy = int(self.vy)
        
        # 6. Round Position
        self.x = round(self.x)
        self.y = round(self.y)

        # Update Timeout
        self.timeout -= 1

def solve_collisions(pods):
    # 4. Resolve Collisions
    # Detecting collisions
    pairs = []
    for i in range(len(pods)):
        for j in range(i + 1, len(pods)):
            p1 = pods[i]
            p2 = pods[j]
            dist2 = p1.distance2(p2)
            if dist2 < (p1.radius + p2.radius)**2:
                pairs.append((p1, p2))

    # Solve for each pair
    # Note: Plan says "Impulse Vector J... New Velocities... Separation".
    # And "Because simulation is discrete, pods might overlap. Move them apart..."
    
    # We need to handle the case where a pod collides with multiple others.
    # The plan for GPU mentions "Scatter-Add", implying simultaneous resolution.
    # The CPU "Truth" should probably handle it sequentially or effectively?
    # Standard Codingame engine is usually sequential but order matters. 
    # Or it processes all impacts based on current state (simultaneous-like).
    # "The collision forces are computed based on the state AT THE MOMENT OF IMPACT (or start of frame if overlap)."
    # Given the discrete nature and simplified plan: "Fixed-Step Iterative Impulse Solver" is for GPU.
    # For CPU, let's implement the standard elastic collision as described.
    
    # We will compute impulses for all overlapping pairs based on CURRENT positions/velocities,
    # then apply them. If we apply sequentially, the second collision uses updated velocity.
    # "Simultaneous" usually means compute all J first, then apply. 
    # Let's stick to the "Truth" definition:
    # "Collisions happen when dist < 2*R... 5. Impulse Vector... 6. New Velocities... 7. Separation"
    
    # Let's do a simple accumulation for now to match the "Scatter-Add" logic concept if possible,
    # OR standard sequential if that's more robust for CPU.
    # Actually, for exact parity with the proposed GPU solver which uses "Scatter-Add",
    # the CPU version should arguably also compute all impulses from the initial state and sum them up?
    # The GPU plan says "Run K=4 iterations... Accumulate... Apply".
    # So CPU should probably do the same loop to ensure parity.
    
    K = 4
    for _ in range(K):
        # Gather all adjustments first?
        # Or simple sequential iterations?
        # GPU Plan: "Compute ... for all pairs simultaneously ... Accumulate ... Apply".
        # This implies we calculate all impulses based on current start-of-iteration state, then apply.
        
        adjustments = {} # Map pod -> (dvx, dvy, dx, dy)
        for p in pods:
            adjustments[p.id] = [0.0, 0.0, 0.0, 0.0]

        pairs_colliding = []
        for i in range(len(pods)):
            for j in range(i + 1, len(pods)):
                p1 = pods[i]
                p2 = pods[j]
                if p1.distance2(p2) < (p1.radius + p2.radius)**2:
                     pairs_colliding.append((p1, p2))
        
        if not pairs_colliding:
            break

        for p1, p2 in pairs_colliding:
            dist = p1.distance(p2)
            if dist == 0: continue # Avoid div by zero

            nx = (p2.x - p1.x) / dist
            ny = (p2.y - p1.y) / dist
            
            dvx = p1.vx - p2.vx
            dvy = p1.vy - p2.vy
            
            # Impact Force F
            # F = (Vrel . N) / (1/m1 + 1/m2)
            product = dvx * nx + dvy * ny
            m1_inv = 1.0 / p1.mass
            m2_inv = 1.0 / p2.mass
            
            f = product / (m1_inv + m2_inv)
            
            # Min Impulse
            if f < MIN_IMPULSE:
                f = MIN_IMPULSE
                
            # Impulse Vector J = -N * F (This looks like scalar F times vector N)
            # Wait, Vrel = V1 - V2.
            # If V1 moving towards V2, Product > 0?
            # V1=(10,0), V2=(0,0). P1=(0,0), P2=(10,0). N=(1,0). Vrel=(10,0). Product=10.
            # J should push P1 back (negative x).
            # J = -N * F = (-1, 0) * 10 = (-10, 0).
            # V1' = V1 + J/m1 = 10 + (-10) = 0. Correct.
            
            jx = -nx * f
            jy = -ny * f
            
            # Separation (Pos Correction)
            # "Move them apart along normal N by (Radius*2 - Dist) / 2 each"
            overlap = (p1.radius + p2.radius) - dist
            sep_mag = overlap / 2.0
            
            # P1 moves opposite to N? No, P1 is at P1, P2 is at P2. N points P1->P2.
            # So P2 should move +N * sep, P1 should move -N * sep.
            
            sx = nx * sep_mag
            sy = ny * sep_mag
            
            # Accumulate
            # P1
            adjustments[p1.id][0] += jx / p1.mass
            adjustments[p1.id][1] += jy / p1.mass
            adjustments[p1.id][2] -= sx
            adjustments[p1.id][3] -= sy
            
            # P2
            adjustments[p2.id][0] -= jx / p2.mass
            adjustments[p2.id][1] -= jy / p2.mass
            adjustments[p2.id][2] += sx
            adjustments[p2.id][3] += sy

        # Apply
        for p in pods:
            adj = adjustments[p.id]
            p.vx += adj[0]
            p.vy += adj[1]
            p.x += adj[2]
            p.y += adj[3]

def step(pods, checkpoints):
    # 1. Rotate
    # 2. Thrust
    # 3. Move
    for p in pods:
        p.apply_movement(checkpoints)
    
    # 4. Resolve Collisions
    solve_collisions(pods)
    
    # 5. Friction & Truncate
    # 6. Round
    for p in pods:
        p.end_frame()
