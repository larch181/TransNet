import cv2
import numpy as np


class Tracker_YOLO:
    def __init__(self):
        # Set up tracker.
        # Instead of MIL, you can also use
        self.bbox = None
        self.old_bbox = np.array([0,0,0,0])
        self.isInited = False
        self.isNeedUpdate = True
        # read pre-trained model and config file
        # net = cv2.dnn.readNet(args.weights, args.config)



        self.init_tracking()

    def init_tracking(self):

        self.tracker = cv2.dnn.readNet('models/detect_model/yolov3-tiny.backup', 'models/detect_model/yolov3-tiny.cfg')
        self.isInited = True
        self.out_layer = self.get_output_layers()

    # in the architecture
    def get_output_layers(self):
        layer_names = self.tracker.getLayerNames()

        output_layers = [layer_names[i[0] - 1] for i in self.tracker.getUnconnectedOutLayers()]

        return output_layers


    def check_update(self):

        diff = np.mean(np.abs(self.old_bbox - self.bbox))
        #print('diff:',diff)
        if diff >= 5:
            self.isNeedUpdate = True
            self.old_bbox = self.bbox.copy()
        else:
            self.isNeedUpdate = False
            self.bbox = self.old_bbox.copy()

    def tracking(self,image):

        Width = image.shape[1]
        Height = image.shape[0]
        scale = 0.00392

        # create input blob
        blob = cv2.dnn.blobFromImage(image, scale, (416,416), (0,0,0), True, crop=False)

        # set input blob for the network
        self.tracker.setInput(blob)

        # run inference through the network
        # and gather predictions from output layers
        outs = self.tracker.forward(self.out_layer)

        # initialization
        class_ids = []
        confidences = []
        boxes = []
        conf_threshold = 0.5
        nms_threshold = 0.4

        # for each detetion from each output layer
        # get the confidence, class id, bounding box params
        # and ignore weak detections (confidence < 0.5)
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5:
                    center_x = int(detection[0] * Width)
                    center_y = int(detection[1] * Height)
                    w = int(detection[2] * Width)
                    h = int(detection[3] * Height)
                    x = center_x - w / 2
                    y = center_y - h / 2
                    class_ids.append(class_id)
                    confidences.append(float(confidence))
                    boxes.append([x, y, w, h])

        # apply non-max suppression
        indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

        # go through the detections remaining
        # after nms and draw bounding box
        self.bbox = np.array(boxes[0])
        #print(self.bbox)
        # Update tracker
        self.check_update()
