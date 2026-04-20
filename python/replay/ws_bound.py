import time
from coppeliasim_zmqremoteapi_client import RemoteAPIClient

# --------------------------------------------------
# WORKSPACE BOUNDARIES
# --------------------------------------------------
X_MIN, X_MAX = -0.18, 0.12
Y_CENTER = 0.546
Y_MIN, Y_MAX = Y_CENTER - 0.10, Y_CENTER + 0.10
Z_HEIGHT = 0.025 

# Define the 4 corners
CORNERS = [
    (X_MIN, Y_MIN), # Back-Left
    (X_MIN, Y_MAX), # Back-Right
    (X_MAX, Y_MAX), # Front-Right
    (X_MAX, Y_MIN)  # Front-Left
]

# --------------------------------------------------
# INITIALIZE
# --------------------------------------------------
client = RemoteAPIClient()
sim = client.getObject('sim')

cube_handle = sim.getObject('/NiryoOne/pick_object')
WORLD = -1 

# --------------------------------------------------
# EXECUTION
# --------------------------------------------------
def test_corners():
    print("🚀 Starting Corner Test...")
    sim.startSimulation()
    
    corner_names = ["Back-Left", "Back-Right", "Front-Right", "Front-Left"]
    
    for i, (cx, cy) in enumerate(CORNERS):
        print(f"📍 Moving to {corner_names[i]}: X={cx:.3f}, Y={cy:.3f}")
        
        # Teleport to corner
        sim.setObjectPosition(cube_handle, WORLD, [cx, cy, Z_HEIGHT])
        
        # Wait 1.5 seconds at each corner for visual check
        time.sleep(1.5)

    # Return to center before stopping
    print(f"🏠 Returning to Center: X=0.00, Y={Y_CENTER}")
    sim.setObjectPosition(cube_handle, WORLD, [0.0, Y_CENTER, Z_HEIGHT])
    time.sleep(1.0)

    print("\n✅ Test complete. Stopping simulation.")
    sim.stopSimulation()

if __name__ == "__main__":
    test_corners()