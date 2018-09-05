import cv2
import sys
import numpy as np
from cv2 import dnn
(major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')


class Tracker:
    def __init__(self):
        # Set up tracker.
        # Instead of MIL, you can also use
        tracker_types = ['BOOSTING', 'MIL', 'KCF', 'TLD', 'MEDIANFLOW', 'GOTURN', 'MOSSE', 'CSRT']
        self.tracker_type = tracker_types[1]
        self.tracker = cv2.TrackerMIL_create()
        self.bbox = None
        self.old_bbox = None
        self.isInited = False
        self.isNeedUpdate = True
    def init_tracking(self,frame,bbox=None):

        # Uncomment the line below to select a different bounding box
        if bbox is None:
            self.bbox = cv2.selectROI(frame, False)
        else:
            self.bbox = bbox

        self.isInited = True

        # Initialize tracker with first frame and bounding box
        ok = self.tracker.init(frame, self.bbox)
        self.bbox = np.array(self.bbox)
        self.old_bbox = np.zeros((1,4))

    def reinit_tracking(self, frame, bbox=None):

        # Uncomment the line below to select a different bounding box
        if bbox is None:
            self.bbox = cv2.selectROI(frame, False)
        else:
            self.bbox = bbox
        self.isInited = True
        self.tracker = cv2.TrackerMIL_create()
        ok = self.tracker.init(frame, self.bbox)
        # Initialize tracker with first frame and bounding box
        self.bbox = np.array(self.bbox)
        self.old_bbox = np.zeros((1, 4))

    def check_update(self):

        diff = np.mean(np.abs(self.old_bbox - self.bbox))

        if diff >= 0.3:
            self.isNeedUpdate = True
            self.old_bbox = self.bbox.copy()
        else:
            self.isNeedUpdate = False
            self.bbox = self.old_bbox.copy()

    def tracking(self,frame):

        # Update tracker
        ok, self.bbox = self.tracker.update(frame)
        self.bbox = np.array(self.bbox)
        self.check_update()

        # # Draw bounding box
        # if ok:
        #     # Tracking success
        #     p1 = (int(self.bbox[0]), int(self.bbox[1]))
        #     p2 = (int(self.bbox[0] + self.bbox[2]), int(self.bbox[1] + self.bbox[3]))
        #     cv2.rectangle(frame, p1, p2, (255, 0, 0), 2, 1)
        # else:
        #     # Tracking failure
        #     cv2.putText(frame, "Tracking failure detected", (100, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
        #
        # # Display tracker type on frame
        # cv2.putText(frame, self.tracker_type + " Tracker", (100, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2)

        # Display result
        #cv2.imshow("Tracking", frame)
