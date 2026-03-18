from coppeliasim_zmqremoteapi_client import RemoteAPIClient
import numpy as np
import cv2
import sys
import csv
import math

# -----------------------------
# Helper Functions
# -----------------------------

def pixel_to_camera(cx, cy, depth, resolution, fov):
    width, height = resolution
    fx = width / (2 * math.tan(fov / 2))
    fy = fx
    x = (cx - width / 2) * depth / fx
    y = (cy - height / 2) * depth / fy
    z = depth
    return np.array([x, y, z])


def camera_to_world(sim, camera, point):
    cam_pos = sim.getObjectPosition(camera, -1)
    cam_ori = sim.getObjectOrientation(camera, -1)
    matrix  = sim.buildMatrix(cam_pos, cam_ori)
    return   sim.multiplyVector(matrix, point)


def get_depth(sim, camera, res):
    try:
        depth_buffer, depth_res = sim.getVisionSensorDepth(camera, 0)

        if isinstance(depth_buffer, (list, tuple)):
            depth = np.array(depth_buffer, dtype=np.float32)
        else:
            depth = np.frombuffer(depth_buffer, dtype=np.float32).copy()

        h = depth_res[1] if depth_res[1] > 0 else res[1]
        w = depth_res[0] if depth_res[0] > 0 else res[0]

        depth = depth.reshape(h, w)
        depth = cv2.flip(depth, 0)
        return depth

    except Exception as e:
        print(f"\nDepth error: {e}")
        return None


# -----------------------------
# Setup
# -----------------------------

client = RemoteAPIClient()
sim    = client.getObject('sim')

sim.setStepping(True)
sim.startSimulation()

camera = sim.getObject('/NiryoOne/camera')

# ✅ Tag for explicit handling BEFORE calling handleVisionSensor
sim.setObjectInt32Param(camera, sim.visionintparam_explicit_handling, 1)
sim.handleVisionSensor(camera)

near = sim.getObjectFloatParam(camera, sim.visionfloatparam_near_clipping)
far  = sim.getObjectFloatParam(camera, sim.visionfloatparam_far_clipping)
fov  = sim.getObjectFloatParam(camera, sim.visionfloatparam_perspective_angle)

print(f"Camera — near: {near:.3f}  far: {far:.3f}  fov: {math.degrees(fov):.1f}°")
print("Recording... Press ESC or Ctrl+C to save and exit.")

pos_log     = []
frame_count = 0

# -----------------------------
# Main Loop
# -----------------------------

try:
    while True:

        sim.step()

        if sim.getSimulationState() == 0:
            break

        # Render fresh frame
        sim.handleVisionSensor(camera)

        # --- Get RGB ---
        try:
            image, resolution = sim.getVisionSensorImg(camera, 0)
        except Exception as e:
            print(f"\nImage error: {e}")
            continue

        if not image:
            continue

        frame_count += 1

        img     = np.frombuffer(image, dtype=np.uint8).reshape(resolution[1], resolution[0], 3)
        img     = cv2.flip(img, 0)
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        # --- Get Depth ---
        depth = get_depth(sim, camera, resolution)
        if depth is None:
            cv2.imshow("camera", img_bgr)
            cv2.waitKey(1)
            continue

        # Uncomment to debug depth values:
        # print(f"\nDepth min:{depth.min():.4f} max:{depth.max():.4f} mean:{depth.mean():.4f}")

        # --- HSV + Red Mask ---
        hsv   = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        mask1 = cv2.inRange(hsv, np.array([0,   70, 50]), np.array([10,  255, 255]))
        mask2 = cv2.inRange(hsv, np.array([170, 70, 50]), np.array([179, 255, 255]))
        mask  = cv2.bitwise_or(mask1, mask2)

        cv2.imshow("mask", mask)

        # --- Contours ---
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            if cv2.contourArea(cnt) < 25:
                continue

            x, y, w, h = cv2.boundingRect(cnt)
            cx = max(0, min(x + w // 2, resolution[0] - 1))
            cy = max(0, min(y + h // 2, resolution[1] - 1))

            z_norm     = depth[cy, cx]
            depth_real = near + z_norm * (far - near)

            cam_point   = pixel_to_camera(cx, cy, depth_real, resolution, fov)
            world_point = camera_to_world(sim, camera, cam_point)
            wx, wy, wz  = world_point[0], world_point[1], world_point[2]

            # Draw
            cv2.rectangle(img_bgr, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.circle(img_bgr, (cx, cy), 5, (0, 0, 255), -1)
            cv2.putText(img_bgr, f"{depth_real:.2f}m", (x, y - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
            cv2.putText(img_bgr, f"({wx:.2f},{wy:.2f},{wz:.2f})", (x, y - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)

            pos_log.append([frame_count, cx, cy, round(depth_real, 4),
                            round(wx, 4), round(wy, 4), round(wz, 4)])

            print(f"Frame {frame_count:04d} | "
                  f"Pixel ({cx:3d},{cy:3d}) | "
                  f"Depth {depth_real:.3f}m | "
                  f"World ({wx:.3f}, {wy:.3f}, {wz:.3f})    ", end="\r")

        cv2.imshow("camera", img_bgr)

        if cv2.waitKey(1) & 0xFF == 27:
            break

except KeyboardInterrupt:
    print("\n\nManual stop.")

finally:
    if pos_log:
        filename = "object_positions.csv"
        with open(filename, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Frame", "Pixel_X", "Pixel_Y", "Depth_m",
                             "World_X", "World_Y", "World_Z"])
            writer.writerows(pos_log)
        print(f"\nSaved {len(pos_log)} positions to '{filename}'")
    else:
        print("\nNo positions recorded.")

    # ✅ Restore sensor to automatic handling before exit
    sim.setObjectInt32Param(camera, sim.visionintparam_explicit_handling, 0)
    sim.setStepping(False)
    sim.stopSimulation()
    cv2.destroyAllWindows()
    sys.exit()