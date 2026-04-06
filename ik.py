# Run this locally to find existing IK dummies
from coppeliasim_zmqremoteapi_client import RemoteAPIClient
client = RemoteAPIClient()
sim = client.getObject('sim')

targets = [
    '/NiryoOne/target',
    '/NiryoOne/IK_target', 
    '/NiryoOne/tip',
    '/NiryoOne/gripper/tip',
    '/NiryoOne/connection/tip',
    '/target',
]

for path in targets:
    try:
        h = sim.getObject(path)
        pos = sim.getObjectPosition(h, -1)
        print(f"✅ {path} → handle {h} pos={pos}")
    except:
        print(f"❌ {path}")