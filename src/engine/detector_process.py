# # detector_process.py
# import multiprocessing
# import time
# import cv2
# import numpy as np
# from src.vision.factory import get_detector


# def run_detector(frame_pipe, det_pipe, person_class_id):
#     """
#     Runs in a separate process.
#     frame_pipe : Pipe connection to receive frames
#     det_pipe   : Pipe connection to send detections (as list of boxes + confidences, or pickled sv.Detections)
#     """
#     # --- create the detector once ---
#     # Import your detector creation here (must be self-contained)
#     detector = get_detector()            # your factory that returns an object with .predict(frame)

#     while True:
#         frame = frame_pipe.recv()
#         if frame is None:
#             break
#         # run detection
#         detections = detector.predict(frame)
#         if detections.class_id is not None:
#             detections = detections[detections.class_id == person_class_id]
#         # send back (we send the xyxy, confidence, class_id as simple arrays, or the whole sv.Detections)
#         # To avoid pickling issues, we can send a tuple: (xyxy, confidence, class_id)
#         if len(detections) == 0:
#             det_pipe.send((None, None, None))   # signal empty
#         else:
#             det_pipe.send((detections.xyxy, detections.confidence, detections.class_id))
#     det_pipe.send(None)   # poison pill