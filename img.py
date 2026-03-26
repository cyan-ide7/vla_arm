from coppeliasim_zmqremoteapi_client import RemoteAPIClient
import numpy as np
import cv2
import sys
import csv # For saving the data

# Setup Client
client = RemoteAPIClient()
sim = client.getObject('sim')

sim.startSimulation()
camera = sim.getObject('/NiryoOne/camera_top')

# List to store our recorded positions: (time_step, x, y)
pos_log = []
frame_count = 0

print("Recording positions... Press Ctrl+C or stop Sim to save and exit.")

try:
    while True:
        if sim.getSimulationState() == 0: 
            break

        image, resolution = sim.getVisionSensorImg(camera, 0)
        if not image:
            continue

        frame_count += 1
        img = np.frombuffer(image, dtype=np.uint8).reshape(resolution[1], resolution[0], 3)
        img = cv2.flip(img, 0)
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

        # Red Mask
        mask1 = cv2.inRange(hsv, np.array([0, 70, 50]), np.array([10, 255, 255]))
        mask2 = cv2.inRange(hsv, np.array([170, 70, 50]), np.array([180, 255, 255]))
        mask = cv2.bitwise_or(mask1, mask2)

        cv2.imshow("mask", mask)

        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        for cnt in contours:
            if cv2.contourArea(cnt) > 10:
                x, y, w, h = cv2.boundingRect(cnt)
                cx, cy = x + w//2, y + h//2
                
                # Draw on image
                cv2.rectangle(img_bgr, (x, y), (x + w, y + h), (0, 255, 0), 2)
                
                #  RECORD DATA
                pos_log.append([frame_count, cx, cy])
                print(f"Recorded Frame {frame_count}: X={cx}, Y={cy}    ", end='\r')

        cv2.imshow("camera", img_bgr)

        if cv2.waitKey(1) & 0xFF == 27:
            break

except KeyboardInterrupt:
    print("\n\nManual stop detected.")

finally:
    # SAVE TO FILE
    if pos_log:
        filename = "object_positions.csv"
        with open(filename, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Frame", "X", "Y"]) # Header
            writer.writerows(pos_log)
        print(f"\nSuccessfully saved {len(pos_log)} positions to {filename}")
    else:
        print("\nNo positions were recorded.")

    sim.stopSimulation()
    cv2.destroyAllWindows()
    sys.exit()