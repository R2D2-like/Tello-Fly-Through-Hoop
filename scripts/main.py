import tello as tello    
import time         
import cv2  
import numpy as np 
from typing import Union
import threading
import datetime
import os
from PIL import Image
from PIL import ImageTk
import tkinter as tki
from tkinter import Toplevel, Scale

WIDTH = 640 # width of the frame
HEIGHT = 480 # height of the frame
DEAD_ZONE = 100 # dead zone of the frame
AREA_MAX_THRESH = 14000 # maximum area of the contour
AREA_MIN_THRESH = 1000 # minimum area of the contour
COUNTER_THRESH = 2 # counter threshold for the drone to fly through the hoop
MATCH_THRESH = 0.1 # match threshold for pattern matching
TARGET_X = WIDTH/2 # target x coordinate
TARGET_Y = 100 # target y coordinate
Kp_X = 25/220 # proportional gain for x coordinate
Kp_Y = 40/300 # proportional gain for y coordinate

class FlyThroughHoop:
    def __init__(self):
        # Initialize the Tello drone
        self.drone = tello.Tello('', 8889)

        # Load the reference image and preprocess it for pattern matching
        img_ref = cv2.imread('hoop.png')
        img_ref = cv2.resize(img_ref, (WIDTH, HEIGHT))
        img_hsv_ref = cv2.cvtColor(img_ref, cv2.COLOR_BGR2HSV_FULL)

        img_h_ref, img_s_ref, img_v_ref = cv2.split(img_hsv_ref)

        ret_ref, thresh_ref = cv2.threshold(img_s_ref, 150, 200, cv2.THRESH_BINARY)
        contours_ref = cv2.findContours(thresh_ref, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)[0]
        self.contours_ref = list(filter(lambda x: cv2.contourArea(x) > AREA_MIN_THRESH, contours_ref))
        
        self.frame = None
        self.distance_counter = 0
        self.stopEvent = threading.Event()
        
        self.distance = 0.1
        self.degree = 30
        
        self.root = tki.Tk()
        self.panel = None

        # Create a button to take a snapshot
        self.btn_snapshot = tki.Button(self.root, text="Snapshot!", command=self.takeSnapshot)
        self.btn_snapshot.pack(side="bottom", fill="both", expand="yes", padx=10, pady=5)
        
        # Create a button to open the command panel
        self.btn_landing = tki.Button(self.root, text="Open Command Panel", relief="raised", command=self.openCmdWindow)
        self.btn_landing.pack(side="bottom", fill="both", expand="yes", padx=10, pady=5)
        
        # Set the title of the window
        self.root.wm_title("TELLO Controller")

        # Set the protocol for the window to close
        self.root.wm_protocol("WM_DELETE_WINDOW", self.onClose)
        
        # Start a thread to read the Tello video stream
        self.video_thread = threading.Thread(target=self.videoLoop)
        self.video_thread.start()
        
        # Start a thread to send command to Tello
        self.command_thread = threading.Thread(target=self._sendingCommand)
        self.command_thread.start()

        self.TRACK_MODE = False

        # Wait for the image from the drone
        try:
            while True:
                frame = self.drone.read()
                if frame is None or frame.size == 0:
                    print('[INFO] No Image')
                    continue
                else:
                    break
        except (KeyboardInterrupt, SystemExit):
            print('[INFO] Interrupted')

        time.sleep(2) # wait for the drone to be ready
        print('[INFO] Initialized')

        self.drone.takeoff() # takeoff the drone

    def videoLoop(self):
        try:
            time.sleep(0.5)
            while not self.stopEvent.is_set():
                self.frame = self.drone.read()
                if self.frame is None or self.frame.size == 0:
                    continue
                image = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)
                image = Image.fromarray(image)
                self.root.after(0, self._updateGUIImage, image)
        except RuntimeError as e:
            print("[INFO] caught a RuntimeError")
            print(e)

    def _updateGUIImage(self, image: Image) -> None:
        image = ImageTk.PhotoImage(image)
        if self.panel is None:
            self.panel = tki.Label(image=image)
            self.panel.image = image
            self.panel.pack(side="left", padx=10, pady=10)
        else:
            self.panel.configure(image=image)
            self.panel.image = image

    def _sendingCommand(self):
        while True:
            self.drone.send_command('command')
            time.sleep(5)

    def openCmdWindow(self):
        panel = Toplevel(self.root)
        panel.wm_title("Command Panel")

        text0 = tki.Label(panel, text='This Controller map keyboard inputs to Tello control commands\nAdjust the trackbar to reset distance and degree parameter', font='Helvetica 10 bold')
        text0.pack(side='top')

        text1 = tki.Label(panel, text='W - Move Tello Up\t\t\tArrow Up - Move Tello Forward\nS - Move Tello Down\t\t\tArrow Down - Move Tello Backward\nA - Rotate Tello Counter-Clockwise\tArrow Left - Move Tello Left\nD - Rotate Tello Clockwise\t\tArrow Right - Move Tello Right', justify="left")
        text1.pack(side="top")

        self.btn_landing = tki.Button(panel, text="Land", relief="raised", command=self.telloLanding)
        self.btn_landing.pack(side="bottom", fill="both", expand="yes", padx=10, pady=5)

        self.btn_takeoff = tki.Button(panel, text="Takeoff", relief="raised", command=self.telloTakeOff)
        self.btn_takeoff.pack(side="bottom", fill="both", expand="yes", padx=10, pady=5)

        self.tmp_f = tki.Frame(panel, width=100, height=2)
        self.tmp_f.bind('<KeyPress-w>', self.on_keypress_w)
        self.tmp_f.bind('<KeyPress-s>', self.on_keypress_s)
        self.tmp_f.bind('<KeyPress-a>', self.on_keypress_a)
        self.tmp_f.bind('<KeyPress-d>', self.on_keypress_d)
        self.tmp_f.bind('<KeyPress-Up>', self.on_keypress_up)
        self.tmp_f.bind('<KeyPress-Down>', self.on_keypress_down)
        self.tmp_f.bind('<KeyPress-Left>', self.on_keypress_left)
        self.tmp_f.bind('<KeyPress-Right>', self.on_keypress_right)
        self.tmp_f.pack(side="bottom")
        self.tmp_f.focus_set()

        self.btn_flip = tki.Button(panel, text="Flip", relief="raised", command=self.openFlipWindow)
        self.btn_flip.pack(side="bottom", fill="both", expand="yes", padx=10, pady=5)

        self.distance_bar = Scale(panel, from_=0.02, to=5, tickinterval=0.01, digits=3, label='Distance(m)', resolution=0.01)
        self.distance_bar.set(0.2)
        self.distance_bar.pack(side="left")

        self.btn_distance = tki.Button(panel, text="Reset Distance", relief="raised", command=self.updateDistancebar)
        self.btn_distance.pack(side="left", fill="both", expand="yes", padx=10, pady=5)

        self.degree_bar = Scale(panel, from_=1, to=360, tickinterval=10, label='Degree')
        self.degree_bar.set(30)
        self.degree_bar.pack(side="right")

        self.btn_degree = tki.Button(panel, text="Reset Degree", relief="raised", command=self.updateDegreebar)
        self.btn_degree.pack(side="right", fill="both", expand="yes", padx=10, pady=5)

    def openFlipWindow(self):
        panel = Toplevel(self.root)
        panel.wm_title("Gesture Recognition")

        self.btn_flipl = tki.Button(panel, text="Flip Left", relief="raised", command=self.telloFlip_l)
        self.btn_flipl.pack(side="bottom", fill="both", expand="yes", padx=10, pady=5)

        self.btn_flipr = tki.Button(panel, text="Flip Right", relief="raised", command=self.telloFlip_r)
        self.btn_flipr.pack(side="bottom", fill="both", expand="yes", padx=10, pady=5)

        self.btn_flipf = tki.Button(panel, text="Flip Forward", relief="raised", command=self.telloFlip_f)
        self.btn_flipf.pack(side="bottom", fill="both", expand="yes", padx=10, pady=5)

        self.btn_flipb = tki.Button(panel, text="Flip Backward", relief="raised", command=self.telloFlip_b)
        self.btn_flipb.pack(side="bottom", fill="both", expand="yes", padx=10, pady=5)

    def takeSnapshot(self):
        ts = datetime.datetime.now()
        filename = "{}.jpg".format(ts.strftime("%Y-%m-%d_%H-%M-%S"))
        p = os.path.sep.join((".", filename))
        cv2.imwrite(p, cv2.cvtColor(self.frame, cv2.COLOR_RGB2BGR))
        print(("[INFO] saved {}".format(filename)))

    def telloTakeOff(self):
        return self.drone.takeoff()

    def telloLanding(self):
        return self.drone.land()

    def telloFlip_l(self):
        return self.drone.flip('l')

    def telloFlip_r(self):
        return self.drone.flip('r')

    def telloFlip_f(self):
        return self.drone.flip('f')

    def telloFlip_b(self):
        return self.drone.flip('b')

    def telloCW(self, degree):
        return self.drone.rotate_cw(degree)

    def telloCCW(self, degree):
        return self.drone.rotate_ccw(degree)

    def telloMoveForward(self, distance):
        return self.drone.move_forward(distance)

    def telloMoveBackward(self, distance):
        return self.drone.move_backward(distance)

    def telloMoveLeft(self, distance):
        return self.drone.move_left(distance)

    def telloMoveRight(self, distance):
        return self.drone.move_right(distance)

    def telloUp(self, dist):
        return self.drone.move_up(dist)

    def telloDown(self, dist):
        return self.drone.move_down(dist)

    def updateDistancebar(self):
        self.distance = self.distance_bar.get()
        print('reset distance to %.1f' % self.distance)

    def updateDegreebar(self):
        self.degree = self.degree_bar.get()
        print('reset degree to %d' % self.degree)

    def on_keypress_w(self, event):
        print("up %d m" % self.distance)
        self.telloUp(self.distance)

    def on_keypress_s(self, event):
        print("down %d m" % self.distance)
        self.telloDown(self.distance)

    def on_keypress_a(self, event):
        print("ccw %d degree" % self.degree)
        self.telloCCW(self.degree)

    def on_keypress_d(self, event):
        print("cw %d m" % self.degree)
        self.telloCW(self.degree)

    def on_keypress_up(self, event):
        print("forward %d m" % self.distance)
        self.telloMoveForward(self.distance)

    def on_keypress_down(self, event):
        print("backward %d m" % self.distance)
        self.telloMoveBackward(self.distance)

    def on_keypress_left(self, event):
        print("left %d m" % self.distance)
        self.telloMoveLeft(self.distance)

    def on_keypress_right(self, event):
        print("right %d m" % self.distance)
        self.telloMoveRight(self.distance)

    def onClose(self):
        print("[INFO] closing...")
        self.stopEvent.set()
        del self.drone
        self.root.quit()

    def getCenter(self, img: np.ndarray) -> Union[list, list]:
        img_raw = img.copy()
        img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV_FULL) # convert the image to HSV color space

        # Extract yellow color 
        lower_color = np.array([20,100,100])
        upper_color = np.array([150,255,255])
        mask = cv2.inRange(img_hsv, lower_color, upper_color)
        yellow_hsv = cv2.bitwise_and(img_hsv, img_hsv, mask = mask)

        img_h, img_s, img_v = cv2.split(yellow_hsv)

        # Extract contours
        ret, thresh = cv2.threshold(img_s, 110, 255, cv2.THRESH_BINARY)
        contours = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)[0]
        contours = list(filter(lambda x: cv2.contourArea(x) > AREA_MIN_THRESH, contours))

        if len(contours) == 0:
            return None, [img_raw, thresh, img_raw]

        # Pattern matching the contours with the reference image
        matched_contours = []
        for cnt in contours:
            ret_match = cv2.matchShapes(self.contours_ref[0], cnt, 1, 0.0)
            if ret_match < MATCH_THRESH:
                area = cv2.contourArea(cnt)
                matched_contours.append({'contour': cnt, 'match_value': ret_match, 'area': area})

        if len(matched_contours) == 0:
            return None, [img_raw, thresh, img_raw]
        
        matched_contours = sorted(matched_contours, key=lambda x: x['match_value'])

        if len(matched_contours) > 2:
            matched_contours = matched_contours[:2]

        best_match_contour = sorted(matched_contours, key=lambda x: x['area'])[0] # always choose the inner contour of the hoop
        
        # Get the center of the contour
        x, y, width, height = cv2.boundingRect(best_match_contour['contour'])
        center = np.array([x + width // 2, y + height // 2])

        # Draw the contour, the center, and the vector to the target
        output = cv2.drawContours(img, best_match_contour['contour'], -1, (0,255,0), 10)
        cv2.circle(output, (center[0], center[1]), radius=5, color=(0, 0, 255), thickness=-1)
        cv2.line(output, (int(TARGET_X), int(TARGET_Y)), (int(center[0]), int(center[1])), (255, 0, 0), 3)
        
        return [center, best_match_contour['area']], [img_raw, thresh, output]

    def action(self, info: list) -> list:
        if info is None:
            if self.TRACK_MODE: # when lost the hoop, wait for the next frame
                return [0, 0, 0, 0]
            return [0, 0, 0, 20] # searching the hoop

        x, y = info[0]
        
        area = info[1]

        # Check if it is close enough to the hoop to fly through
        if area > AREA_MAX_THRESH:
            self.distance_counter += 1
            if self.distance_counter >= COUNTER_THRESH:
                self.distance_counter = 0
                return [0, 35, 0, 0]

        # Proportional control
        if x < TARGET_X:
            yaw = -(TARGET_X - x) * Kp_X
        else:
            yaw = (x - TARGET_X) * Kp_X

        if y < TARGET_Y:
            ud = (TARGET_Y - y) * Kp_Y
        else:
            ud = -(y - TARGET_Y) * Kp_Y     
        
        fb = 20
        lr = 0

        return [int(lr), int(fb), int(ud), int(yaw)]

    def step(self, action: list, imgs: list) -> None:
        if action == [0, 35, 0, 0]: # fly through the hoop
            print('[INFO] Fly through the hoop')
            self.TRACK_MODE = False
            start_time = time.time()
            while time.time() - start_time < 3:
                # Update visualizations
                self.frame = self.drone.read()
                if self.frame is not None and self.frame.size != 0:
                    img = cv2.resize(self.frame, (WIDTH, HEIGHT))
                    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                    _, imgs = self.getCenter(img)
                    self.updateFrame(imgs, 'Fly through the hoop', (255, 0, 0))
                    self.root.update_idletasks()
                    self.root.update()
                # Send command to the drone
                self.drone.set_direction([0, 40, 0, 0])
            return
        
        elif action == [0, 0, 0, 20]: # searching the hoop
            # Update visualizations
            text = 'Searching'
            print('[INFO] Searching')
            color = (0, 255, 0)
            self.updateFrame(imgs, text, color)
            self.root.update_idletasks()
            self.root.update()
            # Send command to the drone
            self.drone.set_direction(action)
            return
        
        else: # tracking the hoop
            print('[INFO] Tracking')
            if not self.TRACK_MODE:
                self.TRACK_MODE = True
            # Update visualizations
            text = 'Tracking'
            color = (0, 0, 255)
            self.updateFrame(imgs, text, color)
            self.root.update_idletasks()
            self.root.update()
            # Send command to the drone
            self.drone.set_direction(action)
            return

    def updateFrame(self, imgs: list, text: str, color: tuple) -> None:
        output = cv2.putText(imgs[2], text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color , 2, cv2.LINE_AA)
        stack = self.stackImages(0.9, ([imgs[0], imgs[1], output]))
        image = cv2.cvtColor(stack, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
        self.root.after(0, self._updateGUIImage, image)

    def stackImages(self, scale: float, imgs: list) -> np.ndarray:
        rows = len(imgs)
        cols = len(imgs[0]) if isinstance(imgs[0], list) else 1
        width = imgs[0][0].shape[1] if cols > 1 else imgs[0].shape[1]
        height = imgs[0][0].shape[0] if cols > 1 else imgs[0].shape[0]

        resized_imgs = []
        for row in imgs:
            resized_row = []
            for img in (row if cols > 1 else [row]):
                if img.shape[:2] != (height, width):
                    img = cv2.resize(img, (width, height))
                if img.shape[:2] != (height, width):
                    img = cv2.resize(img, (width, height))
                if len(img.shape) == 2:
                    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                img = cv2.resize(img, (0, 0), None, scale, scale)
                resized_row.append(img)
            resized_imgs.append(resized_row if cols > 1 else resized_row[0])

        imageBlank = np.zeros((int(height * scale), int(width * scale), 3), np.uint8)
        hor = [imageBlank] * rows
        for x in range(rows):
            hor[x] = np.hstack(resized_imgs[x]) if cols > 1 else resized_imgs[x]
        ver = np.vstack(hor)
        return ver

def main() -> None:
    task = FlyThroughHoop()

    try:
        while True:
            frame = task.drone.read()
            if frame is None or frame.size == 0: # check if the frame is ready
                print('[INFO] No Image')
                continue
            
            # Preprocess the frame
            img = cv2.resize(frame, (WIDTH, HEIGHT))
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

            # Get the center of the hoop
            info, imgs = task.getCenter(img)

            # Determine the action
            action = task.action(info)

            # Take the action
            task.step(action, imgs)

            # Update visualizations
            task.root.update_idletasks()
            task.root.update()

    except (KeyboardInterrupt, SystemExit): # stop the drone and close the window
        cv2.destroyAllWindows()
        task.drone.land()
        del task

if __name__ == "__main__":
    main()
