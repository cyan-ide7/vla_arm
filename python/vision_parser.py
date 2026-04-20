"""
vision_parser.py
================
Fully local object detection using YOLOv8 + colour filtering.
No API key. No usage limits. No internet required after first model download.

How it works:
    1. Grab the top-down camera frame from CoppeliaSim
    2. Run YOLOv8 to get bounding boxes + class labels
    3. Filter detections by colour (HSV mask) and label keyword from user text
    4. Convert the 2D bounding box centre to a 3D world position using
       the known table height and camera intrinsics
    5. Return a 6D goal pose [x, y, z, pitch, yaw, gripper]

Setup:
    pip install ultralytics opencv-python numpy

The YOLOv8n weights (~6 MB) download automatically on first run.
For custom objects (your specific cubes/cylinders) you can train a tiny
YOLO model later — but the default COCO weights already detect
"cup", "bottle", "bowl", "sports ball" which covers most lab objects.
"""

import re
import numpy as np
import cv2
from ultralytics import YOLO

# --------------------------------------------------
# CAMERA INTRINSICS
# Match these to your CoppeliaSim vision sensor settings.
# In CoppeliaSim:
#   sensor properties → perspective angle (FOV in degrees)
#   resolution (width x height)
# --------------------------------------------------
CAM_WIDTH      = 256         # pixels
CAM_HEIGHT     = 256          # pixels
CAM_FOV_DEG    = 60.0         # perspective angle (degrees)
CAM_HEIGHT_M   = 0.75         # metres above table surface

# Derived focal length (pixels)
CAM_FOV_RAD    = np.deg2rad(CAM_FOV_DEG)
FOCAL_LENGTH   = (CAM_WIDTH / 2.0) / np.tan(CAM_FOV_RAD / 2.0)

# Camera optical centre
CX = CAM_WIDTH  / 2.0
CY = CAM_HEIGHT / 2.0

# --------------------------------------------------
# WORKSPACE
# --------------------------------------------------
TABLE_Z       = 0          # metres
PICK_PITCH    = -1.2          # ~-70 deg (wrist down for grasp)
PLACE_PITCH   = -1.2
GRIPPER_OPEN  = 0.0
GRIPPER_CLOSE = 1.0

# Named place zones — user says "put it on the left", "drop it in zone b"
ZONES: dict[str, tuple] = {
    "left":       (0.35,  0.25, TABLE_Z),
    "right":      (0.35, -0.25, TABLE_Z),
    "center":     (0.40,  0.00, TABLE_Z),
    "middle":     (0.40,  0.00, TABLE_Z),
    "front":      (0.30,  0.00, TABLE_Z),
    "back":       (0.55,  0.00, TABLE_Z),
    "left side":  (0.35,  0.25, TABLE_Z),
    "right side": (0.35, -0.25, TABLE_Z),
    "zone a":     (0.50,  0.20, TABLE_Z),
    "zone b":     (0.50, -0.20, TABLE_Z),
    "zone c":     (0.60,  0.00, TABLE_Z),
    "shelf":      (0.65,  0.00, TABLE_Z + 0.10),
    "tray":       (0.45,  0.30, TABLE_Z),
    "bin":        (0.20,  0.35, TABLE_Z),
    "corner":     (0.60,  0.30, TABLE_Z),
    "home":       (0.40,  0.00, TABLE_Z),
}

RELATIVE_OFFSETS: dict[str, tuple] = {
    "next to":     ( 0.00,  0.12, 0.00),
    "beside":      ( 0.00,  0.12, 0.00),
    "left of":     ( 0.00,  0.14, 0.00),
    "right of":    ( 0.00, -0.14, 0.00),
    "behind":      (-0.12,  0.00, 0.00),
    "in front of": ( 0.12,  0.00, 0.00),
    "on top of":   ( 0.00,  0.00, 0.08),
    "on":          ( 0.00,  0.00, 0.08),
    "near":        ( 0.00,  0.12, 0.00),
}

# --------------------------------------------------
# COLOUR DETECTION  (HSV ranges)
# Each entry: colour_name → (lower_hsv, upper_hsv)
# Tune these if your sim colours look different.
# --------------------------------------------------
COLOUR_RANGES = {
    "red": [
        (np.array([  0, 120,  70]), np.array([ 10, 255, 255])),
        (np.array([170, 120,  70]), np.array([180, 255, 255])),   # red wraps in HSV
    ],
    "green":  [(np.array([ 36,  80,  40]), np.array([ 86, 255, 255]))],
    "blue":   [(np.array([100,  80,  40]), np.array([130, 255, 255]))],
    "yellow": [(np.array([ 20, 100,  80]), np.array([ 35, 255, 255]))],
    "purple": [(np.array([130,  50,  40]), np.array([160, 255, 255]))],
    "orange": [(np.array([ 10, 100,  80]), np.array([ 20, 255, 255]))],
    "white":  [(np.array([  0,   0, 180]), np.array([180,  40, 255]))],
    "black":  [(np.array([  0,   0,   0]), np.array([180, 255,  50]))],
}

# YOLO COCO labels that correspond to "objects on a table"
# Add more if you train a custom model later
OBJECT_LABELS = {
    "cup", "bottle", "bowl", "sports ball", "orange", "apple",
    "banana", "donut", "cake", "book", "cell phone", "remote",
    "scissors", "teddy bear", "vase", "clock",
    # custom labels if you train your own model:
    "cube", "cylinder", "sphere", "box", "block",
}

# --------------------------------------------------
# LOAD YOLO MODEL
# Downloads ~6 MB yolov8n.pt on first run, then cached locally.
# Switch to "yolov8s.pt" or "yolov8m.pt" for better accuracy.
# --------------------------------------------------
print("[Vision] Loading YOLOv8n...")
_yolo = YOLO("yolov8n.pt")
print("[Vision] YOLOv8n ready.")


# --------------------------------------------------
# TEXT PARSING  (pure Python — no LLM)
# Extracts colour + shape + place location from user text.
# --------------------------------------------------
COLOUR_KEYWORDS = list(COLOUR_RANGES.keys())
SHAPE_KEYWORDS  = ["cube", "cylinder", "sphere", "box", "ball",
                   "block", "object", "thing", "item", "piece"]
PLACE_KEYWORDS  = list(ZONES.keys()) + list(RELATIVE_OFFSETS.keys())


def _parse_user_text(user_text: str) -> dict:
    """
    Extract intent from user text without any LLM.
    Scans for colour, shape, and place keywords.
    """
    text = user_text.lower().strip()

    # Colour
    pick_colour = "any"
    for c in COLOUR_KEYWORDS:
        if c in text:
            pick_colour = c
            break

    # Shape
    pick_shape = "any"
    for s in SHAPE_KEYWORDS:
        if s in text:
            pick_shape = s
            break

    # Place location — try multi-word zones first (longest match wins)
    place_location = "center"
    for zone in sorted(ZONES.keys(), key=len, reverse=True):
        if zone in text:
            place_location = zone
            break

    # Relative placement — e.g. "next to the green cube"
    place_relative_to = ""
    place_relation    = ""
    for rel in sorted(RELATIVE_OFFSETS.keys(), key=len, reverse=True):
        if rel in text:
            place_relation = rel
            # try to find what object it is relative to
            after = text.split(rel, 1)[-1]
            for c in COLOUR_KEYWORDS:
                if c in after:
                    place_relative_to = c
                    break
            break

    print(f"[Vision] Parsed: colour={pick_colour}  shape={pick_shape}  "
          f"place='{place_location}'  rel='{place_relation}' '{place_relative_to}'")

    return {
        "pick_colour":       pick_colour,
        "pick_shape":        pick_shape,
        "place_location":    place_location,
        "place_relative_to": place_relative_to,
        "place_relation":    place_relation,
    }


# --------------------------------------------------
# COLOUR MASKING
# --------------------------------------------------
def _colour_score(roi_bgr: np.ndarray, colour: str) -> float:
    """
    Returns fraction of pixels in roi_bgr that match the given colour.
    0.0 = no match, 1.0 = perfect match.
    """
    if colour == "any" or colour not in COLOUR_RANGES:
        return 1.0   # accept any colour

    hsv   = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
    mask  = np.zeros(hsv.shape[:2], dtype=np.uint8)

    for (lo, hi) in COLOUR_RANGES[colour]:
        mask |= cv2.inRange(hsv, lo, hi)

    total = mask.size
    return float(np.count_nonzero(mask)) / total if total > 0 else 0.0


# --------------------------------------------------
# 2D BOUNDING BOX CENTRE → 3D WORLD POSITION
# --------------------------------------------------
def _bbox_to_world(cx_px: float, cy_px: float,
                   object_z: float = TABLE_Z + 0.03) -> tuple:
    """
    Convert a 2D pixel coordinate (bbox centre) to a 3D world position.

    Assumes:
        - Camera is mounted directly above the table, pointing straight down
        - Camera height above the table surface = CAM_HEIGHT_M
        - Simple pinhole camera model

    Returns:
        (x, y, z) in world metres
    """
    # Distance from camera to object surface
    depth = CAM_HEIGHT_M - (object_z - TABLE_Z)

    # Pixel offset from image centre
    dx_px = cx_px - CX
    dy_px = cy_px - CY

    # Convert to metres using pinhole model
    x_cam =  dx_px * depth / FOCAL_LENGTH
    y_cam =  dy_px * depth / FOCAL_LENGTH

    # Camera frame → world frame
    # Assumes camera X = world X (forward), camera Y = world -Y (left)
    # Adjust signs/axes if your camera is rotated differently
    world_x = (TABLE_Z + CAM_HEIGHT_M) * 0.5 + x_cam   # approximate table centre offset
    world_y = -y_cam
    world_z = object_z

    return (world_x, world_y, world_z)


def _bbox_centre_to_world(x1: float, y1: float,
                           x2: float, y2: float) -> tuple:
    """Convenience: takes bbox corners, returns world (x, y, z)."""
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    return _bbox_to_world(cx, cy)


# --------------------------------------------------
# YOLO DETECTION ON A FRAME
# --------------------------------------------------
def detect_objects(frame_bgr: np.ndarray,
                   target_colour: str = "any",
                   target_shape:  str = "any",
                   conf_threshold: float = 0.30,
                   colour_threshold: float = 0.15,
                   debug_window: bool = False
                   ) -> list[dict]:
    """
    Run YOLOv8 on frame_bgr and return filtered detections.

    Each detection dict:
        {
          "label":      str,       YOLO class name
          "conf":       float,     detection confidence
          "bbox":       (x1,y1,x2,y2),  pixels
          "centre_px":  (cx, cy),  pixels
          "world_pos":  (x, y, z), metres
          "colour_score": float
        }

    Args:
        frame_bgr       OpenCV BGR image
        target_colour   colour keyword to filter by (or "any")
        target_shape    shape/label keyword to filter by (or "any")
        conf_threshold  minimum YOLO confidence
        colour_threshold minimum colour mask fraction to accept
        debug_window    if True, shows a window with drawn boxes
    """
    results = _yolo(frame_bgr, verbose=False)[0]
    detections = []

    for box in results.boxes:
        conf  = float(box.conf[0])
        if conf < conf_threshold:
            continue

        label = results.names[int(box.cls[0])].lower()

        # Label filter — skip if shape specified and label doesn't match
        if target_shape != "any":
            if target_shape not in label and label not in OBJECT_LABELS:
                continue

        x1, y1, x2, y2 = map(float, box.xyxy[0])

        # Crop ROI for colour check
        x1i, y1i = int(max(0, x1)), int(max(0, y1))
        x2i, y2i = int(min(frame_bgr.shape[1], x2)), int(min(frame_bgr.shape[0], y2))
        roi = frame_bgr[y1i:y2i, x1i:x2i]

        if roi.size == 0:
            continue

        cscore = _colour_score(roi, target_colour)
        if cscore < colour_threshold:
            continue

        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        world = _bbox_centre_to_world(x1, y1, x2, y2)

        detections.append({
            "label":        label,
            "conf":         conf,
            "bbox":         (x1, y1, x2, y2),
            "centre_px":    (cx, cy),
            "world_pos":    world,
            "colour_score": cscore,
        })

    # Sort: highest colour_score × conf first
    detections.sort(key=lambda d: d["colour_score"] * d["conf"], reverse=True)

    if debug_window and len(detections) > 0:
        vis = frame_bgr.copy()
        for d in detections:
            x1, y1, x2, y2 = map(int, d["bbox"])
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
            txt = f"{d['label']} {d['conf']:.2f} col={d['colour_score']:.2f}"
            cv2.putText(vis, txt, (x1, y1 - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
        cv2.imshow("YOLO detections", vis)
        cv2.waitKey(1)

    return detections


# --------------------------------------------------
# PLACE POSE RESOLVER  (same logic as gemini_parser)
# --------------------------------------------------
def _resolve_place_pose(intent: dict,
                         pick_world: tuple) -> np.ndarray:
    location    = intent.get("place_location",   "center").lower()
    relative_to = intent.get("place_relative_to","").lower()
    relation    = intent.get("place_relation",   "").lower()

    # Named zone
    for zone_key, pos in sorted(ZONES.items(), key=lambda kv: -len(kv[0])):
        if zone_key in location:
            return np.array([pos[0], pos[1], pos[2],
                             PLACE_PITCH, 0.0, GRIPPER_OPEN], dtype=np.float32)

    # Relative to another colour — uses ZONES fallback since we can't run
    # YOLO twice; real-time impl should look up the reference object live
    if relative_to and relative_to in ZONES:
        ref_pos = ZONES[relative_to]
        offset  = RELATIVE_OFFSETS.get(relation, (0.00, 0.12, 0.00))
        return np.array([ref_pos[0] + offset[0],
                         ref_pos[1] + offset[1],
                         ref_pos[2] + offset[2],
                         PLACE_PITCH, 0.0, GRIPPER_OPEN], dtype=np.float32)

    # Inline relative keyword
    for rel_kw, offset in sorted(RELATIVE_OFFSETS.items(), key=lambda kv: -len(kv[0])):
        if rel_kw in location:
            px, py, pz = pick_world
            return np.array([px + offset[0], py + offset[1], pz + offset[2],
                             PLACE_PITCH, 0.0, GRIPPER_OPEN], dtype=np.float32)

    # Fallback: beside pick position
    px, py, pz = pick_world
    return np.array([px, py + 0.15, pz,
                     PLACE_PITCH, 0.0, GRIPPER_OPEN], dtype=np.float32)


# --------------------------------------------------
# PUBLIC API
# --------------------------------------------------
def parse_goal_pose_from_frame(user_text: str,
                                frame_bgr: np.ndarray,
                                task: str = "place",
                                debug: bool = False
                                ) -> np.ndarray:
    """
    Main entry point for real-time visual parsing.

    Args:
        user_text   Natural language command
        frame_bgr   Current top-camera frame (BGR, from CoppeliaSim)
        task        "place" → return place pose (default)
                    "pick"  → return pick pose
        debug       Show detection window

    Returns:
        np.ndarray [6] = [x, y, z, pitch, yaw, gripper]  in world metres
        NOT yet normalised — call normalize_action() in vla_interface.py
    """
    print(f"\n[Vision] Command: '{user_text}'")
    intent = _parse_user_text(user_text)

    detections = detect_objects(
        frame_bgr,
        target_colour=intent["pick_colour"],
        target_shape=intent["pick_shape"],
        debug_window=debug
    )

    if not detections:
        print("[Vision] No matching object detected. Defaulting to table center.")
        pick_world = (0.40, 0.00, TABLE_Z + 0.03)
    else:
        best = detections[0]
        pick_world = best["world_pos"]
        print(f"[Vision] Best match: label='{best['label']}'  "
              f"conf={best['conf']:.2f}  colour_score={best['colour_score']:.2f}")
        print(f"[Vision] Pixel centre: {best['centre_px']}")
        print(f"[Vision] World pos:    {pick_world}")

    pick_pose = np.array([
        pick_world[0], pick_world[1], pick_world[2],
        PICK_PITCH, 0.0, GRIPPER_CLOSE
    ], dtype=np.float32)

    place_pose = _resolve_place_pose(intent, pick_world)

    print(f"[Vision] Pick  → {pick_pose}")
    print(f"[Vision] Place → {place_pose}")

    return place_pose if task == "place" else pick_pose


def parse_goal_pose(user_text: str,
                    frame_bgr: np.ndarray = None,
                    task: str = "place") -> np.ndarray:
    """
    Alias so vla_interface.py import stays identical:
        from vision_parser import parse_goal_pose
    If no frame provided, returns center table position.
    """
    if frame_bgr is None:
        print("[Vision] No frame provided — returning table center.")
        return np.array([0.40, 0.00, TABLE_Z, PLACE_PITCH, 0.0, GRIPPER_OPEN],
                        dtype=np.float32)
    return parse_goal_pose_from_frame(user_text, frame_bgr, task)


# --------------------------------------------------
# SELF-TEST  (uses webcam or a test image)
# --------------------------------------------------
if __name__ == "__main__":
    import sys

    print("=" * 56)
    print("  Vision Parser — self test")
    print("=" * 56)

    # Try to open webcam, fall back to blank frame
    cap = cv2.VideoCapture(0)
    if cap.isOpened():
        ret, frame = cap.read()
        cap.release()
        if not ret:
            frame = np.zeros((512, 512, 3), dtype=np.uint8)
            print("[Test] Webcam read failed, using blank frame.")
        else:
            print("[Test] Using webcam frame.")
    else:
        frame = np.zeros((512, 512, 3), dtype=np.uint8)
        print("[Test] No webcam found, using blank frame.")

    # Draw some coloured shapes on the blank frame to test detection
    if np.all(frame == 0):
        cv2.rectangle(frame, (100, 150), (180, 230), (0,   0, 220), -1)   # red cube area
        cv2.rectangle(frame, (280, 150), (360, 230), (0, 200,  50), -1)   # green cube area
        cv2.circle(frame,    (420, 190), 40,         (220, 50,  0), -1)   # blue circle
        print("[Test] Drew synthetic coloured shapes on blank frame.")

    tests = [
        ("pick the red cube and put it on the left",  "place"),
        ("move the green object to zone b",           "place"),
        ("grab the blue thing and put it in the bin", "place"),
    ]

    for cmd, task in tests:
        pose = parse_goal_pose_from_frame(cmd, frame, task=task, debug=True)
        print(f"  → x={pose[0]:.3f}  y={pose[1]:.3f}  z={pose[2]:.3f}  "
              f"gripper={pose[5]:.1f}\n")

    cv2.destroyAllWindows()