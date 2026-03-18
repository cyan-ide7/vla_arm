from coppeliasim_zmqremoteapi_client import RemoteAPIClient
import cv2
import numpy as np

client = RemoteAPIClient()
sim = client.getObject('sim')

# start simulation
sim.startSimulation()

# get camera handle
camera = sim.getObject('/NiryoOne/camera')

while True:
    
    image, resolution = sim.getVisionSensorImg(camera)

    img = np.frombuffer(image, dtype=np.uint8)
    img = img.reshape(resolution[1], resolution[0], 3)

    img = cv2.flip(img, 0)

    cv2.imshow("Camera", img)

    if cv2.waitKey(1) == 27:
        break

sim.stopSimulation