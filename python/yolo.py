import time
import numpy as np
import cv2
from coppeliasim_zmqremoteapi_client import RemoteAPIClient

# --------------------------------------------------
# INITIALIZE
# --------------------------------------------------
client = RemoteAPIClient()
sim = client.getObject('sim')
sim.startSimulation()

niryo_handle = sim.getObject('/NiryoOne')
script_handle = sim.getScript(sim.scripttype_childscript, niryo_handle)

# --- REFINED COLOR THRESHOLDS ---
COLORS = {
    # Red is split into two ranges because it sits at both ends of the HSV Hue cylinder
    "RED":   [(np.array([0, 150, 50]), np.array([10, 255, 255])), 
              (np.array([160, 150, 50]), np.array([180, 255, 255]))],
    "GREEN": [(np.array([35, 100, 50]), np.array([90, 255, 255]))],
    "WHITE": [(np.array([0, 0, 240]), np.array([180, 25, 255]))], 
    "BLACK": [(np.array([0, 0, 0]), np.array([180, 255, 45]))]
}

def convert_image(img_buffer, resolution):
    width, height = resolution
    img = np.frombuffer(img_buffer, dtype=np.uint8).reshape(height, width, 3)
    return cv2.cvtColor(np.flipud(img), cv2.COLOR_RGB2BGR)

def detect_shape(contour):
    perimeter = cv2.arcLength(contour, True)
    # Using a smaller factor (0.05) to keep the shape detection sensitive for small objects
    approx = cv2.approxPolyDP(contour, 0.05 * perimeter, True)
    
    if len(approx) == 4: 
        return "Square"
    elif len(approx) > 4: 
        return "Circle"
    return "Object"

try:
    print("Detecting all objects... Press ESC to quit.")
    while True:
        obs = sim.callScriptFunction('getObservation', script_handle, [])
        top_img = convert_image(obs['top_image'], obs['top_resolution'])
        h, w = top_img.shape[:2]

        # Masking the arm area (bottom of the screen)
        proc_img = top_img.copy()
        proc_img[int(h*0.75):h, :] = (0, 0, 0) 
        
        hsv = cv2.cvtColor(proc_img, cv2.COLOR_BGR2HSV)

        for color_name, ranges in COLORS.items():
            mask = np.zeros(hsv.shape[:2], dtype="uint8")
            for (low, high) in ranges:
                mask = cv2.bitwise_or(mask, cv2.inRange(hsv, low, high))

            # Use a very small kernel (3x3) to clean noise without erasing the red square
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3,3), np.uint8))
            cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for c in cnts:
                area = cv2.contourArea(c)
                
                # CRITICAL: Lowered to 20 pixels to catch that tiny red square
                if area > 20: 
                    shape = detect_shape(c)
                    x, y, w_box, h_box = cv2.boundingRect(c)
                    
                    color_map = {
                        "RED":(0,0,255), "GREEN":(0,255,0), 
                        "BLACK":(0,0,0), "WHITE":(255,255,255)
                    }
                    draw_col = color_map.get(color_name, (0,255,0))

                    # Labeling logic
                    label = f"{color_name} {shape}"
                    if area < 100: label = f"Small {label}"

                    cv2.rectangle(top_img, (x, y), (x+w_box, y+h_box), draw_col, 2)
                    cv2.putText(top_img, label, (x, y-10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, draw_col, 1)

        cv2.imshow("Filtered Detection", top_img)
        if cv2.waitKey(1) & 0xFF == 27: break

finally:
    sim.stopSimulation()
    cv2.destroyAllWindows()