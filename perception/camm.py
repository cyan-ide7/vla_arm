from coppeliasim_zmqremoteapi_client import RemoteAPIClient
client = RemoteAPIClient()
sim = client.getObject('sim')

# Try all possible camera paths from your scene
paths = [
    '/camera_vla',
    'camera_vla',
    '/vla_rm/camera_vla',
    '/NiryoOne/camera_vla',
    '/camera_top',
    '/vision_side',
    '/vision_front',
]

for path in paths:
    try:
        handle = sim.getObject(path)
        print(f"✅ FOUND: {path} → handle {handle}")
    except:
        print(f"❌ Not found: {path}")