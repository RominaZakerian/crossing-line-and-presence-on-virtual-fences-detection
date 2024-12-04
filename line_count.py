
import datetime
import argparse
from collections import defaultdict
from pathlib import Path
from numpy import ones,vstack
from numpy.linalg import lstsq
import cv2
import numpy as np
from shapely.geometry import Polygon,LineString
from shapely.geometry.point import Point
from time import sleep
from ultralytics import YOLO
from ultralytics.utils.files import increment_path
from ultralytics.utils.plotting import Annotator, colors
import json

# Open and read the JSON file
with open('data/policy.json', 'r') as file:
    data = json.load(file)

# Print the data
print(data)


track_history = defaultdict(list)

current_region = None
counting_regions = [
    {
        "name": "fance 1",
        "polygon": LineString(data["lines"][0]),  # Polygon points
        "counts": 0,
        "dragging": False,
        "region_color": (0, 255, 0),  # BGR Value
        "text_color": (0, 0, 0),  # Region Text Color
    },
    {
        "name": "YOLOv8 Rectangle Region",
        "polygon": LineString(data["lines"][1]),  # Polygon points
        "counts": 0,
        "dragging": False,
        "region_color": (0, 255, 0),  # BGR Value
        "text_color": (0, 0, 0),  # Region Text Color
    },
]


points1 = [(586,502),(170,280)]
x_coords1, y_coords1 = zip(*points1)
A1 = vstack([x_coords1,ones(len(x_coords1))]).T
m1, c1 = lstsq(A1, y_coords1)[0]
print("Line Solution is y = {m}x + {c}".format(m=m1,c=c1))


points2 = [(644,481),(1258,297)]
x_coords2, y_coords2 = zip(*points2)
A2 = vstack([x_coords2,ones(len(x_coords2))]).T
m2, c2 = lstsq(A2, y_coords2)[0]
print("Line Solution is y = {m}x + {c}".format(m=m2,c=c2))

def mouse_callback(event, x, y, flags, param):
    """
    Handles mouse events for region manipulation.

    Args:
        event (int): The mouse event type (e.g., cv2.EVENT_LBUTTONDOWN).
        x (int): The x-coordinate of the mouse pointer.
        y (int): The y-coordinate of the mouse pointer.
        flags (int): Additional flags passed by OpenCV.
        param: Additional parameters passed to the callback (not used in this function).

    Global Variables:
        current_region (dict): A dictionary representing the current selected region.

    Mouse Events:
        - LBUTTONDOWN: Initiates dragging for the region containing the clicked point.
        - MOUSEMOVE: Moves the selected region if dragging is active.
        - LBUTTONUP: Ends dragging for the selected region.

    Notes:
        - This function is intended to be used as a callback for OpenCV mouse events.
        - Requires the existence of the 'counting_regions' list and the 'Polygon' class.

    Example:
        >>> cv2.setMouseCallback(window_name, mouse_callback)
    """
    global current_region

    # Mouse left button down event
    if event == cv2.EVENT_LBUTTONDOWN:
        for region in counting_regions:
            if region["polygon"].contains(Point((x, y))):
                current_region = region
                current_region["dragging"] = True
                current_region["offset_x"] = x
                current_region["offset_y"] = y

    # Mouse move event
    elif event == cv2.EVENT_MOUSEMOVE:
        if current_region is not None and current_region["dragging"]:
            dx = x - current_region["offset_x"]
            dy = y - current_region["offset_y"]
            current_region["polygon"] = Polygon(
                [(p[0] + dx, p[1] + dy) for p in current_region["polygon"].exterior.coords]
            )
            current_region["offset_x"] = x
            current_region["offset_y"] = y

    # Mouse left button up event
    elif event == cv2.EVENT_LBUTTONUP:
        if current_region is not None and current_region["dragging"]:
            current_region["dragging"] = False


def run(
    weights="yolov8n.pt",
    source=None,
    device="cpu",
    view_img=False,
    save_img=False,
    exist_ok=False,
    classes=None,
    line_thickness=2,
    track_thickness=2,
    region_thickness=2,
):
    """
    Run Region counting on a video using YOLOv8 and ByteTrack.

    Supports movable region for real time counting inside specific area.
    Supports multiple regions counting.
    Regions can be Polygons or rectangle in shape

    Args:
        weights (str): Model weights path.
        source (str): Video file path.
        device (str): processing device cpu, 0, 1
        view_img (bool): Show results.
        save_img (bool): Save results.
        exist_ok (bool): Overwrite existing files.
        classes (list): classes to detect and track
        line_thickness (int): Bounding box thickness.
        track_thickness (int): Tracking line thickness
        region_thickness (int): Region thickness.
    """
    vid_frame_count = 0

    # Check source path
    if not Path(source).exists():
        raise FileNotFoundError(f"Source path '{source}' does not exist.")

    # Setup Model
    model = YOLO(f"{weights}")
    model.to("cpu")

    # Extract classes names
    names = model.model.names

    # Video setup
    videocapture = cv2.VideoCapture(source)
    frame_width, frame_height = int(videocapture.get(3)), int(videocapture.get(4))
    fps, fourcc = int(videocapture.get(5)), cv2.VideoWriter_fourcc(*"mp4v")

    # Output setup
    save_dir = Path("./data/")
    save_dir.mkdir(parents=True, exist_ok=True)
    video_writer = cv2.VideoWriter(str(save_dir / "output_lines.mp4"), fourcc, fps, (frame_width, frame_height))
    
    # Iterate over video frames
   
    counter1 = 0
    counter2=0
    color_change1 = 0
    color_change2 = 0
    while videocapture.isOpened():
        start = datetime.datetime.now()
        success, frame = videocapture.read()
        if not success:
            break
        vid_frame_count += 1
        

        # Extract the results
        results = model.track(frame, persist=True, classes=classes,conf=0.4, iou=0.5,  tracker="bytetrack.yaml")

        # Draw regions (Polygons/Rectangles)
        for region in counting_regions:
            region_label = str(region["counts"])
            region_color = region["region_color"]
            region_text_color = region["text_color"]

            polygon_coords = np.array(region["polygon"].coords, dtype=np.int32)

            text_size, _ = cv2.getTextSize(
                region_label, cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.7, thickness=line_thickness)
            
            cv2.line(frame, polygon_coords[0],polygon_coords[1], color=region_color, thickness=region_thickness)
            cv2.rectangle(
                frame,
                (polygon_coords[0][0] - 5, polygon_coords[0][1] - text_size[1] - 5),
                (polygon_coords[0][0] + text_size[0] + 5, polygon_coords[0][1] + 5),
                region_color,
                -1,)
            cv2.putText(
                frame, region_label, (polygon_coords[0][0] ,polygon_coords[0][1]), cv2.FONT_HERSHEY_SIMPLEX, 0.7, region_text_color, line_thickness)
            


        if results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu()
            track_ids = results[0].boxes.id.int().cpu().tolist()
            clss = results[0].boxes.cls.cpu().tolist()

            frame = results[0].plot()

            annotator = Annotator(frame, line_width=line_thickness, example=str(names))

            for box, track_id, cls in zip(boxes, track_ids, clss):
                # annotator.box_label(box, str(names[cls]), color=colors(cls, True))
                bbox_center = (box[0] + box[2]) /2 , ((box[1] + box[3]) / 2)
                if cls == 0:
                    bbox_center = (box[0] + box[2]) /2 , 2*((box[1] + box[3]) / 3)  # Bbox center

                track = track_history[track_id]  # Tracking Lines plot
                track.append((float(bbox_center[0]), float(bbox_center[1])))
                if len(track) > 30:
                    track.pop(0)
                points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
                cv2.polylines(frame, [points], isClosed=False, color=colors(cls, True), thickness=track_thickness)

                # Check if detection inside region
                for region in counting_regions:
                   
                    region_color = region["region_color"]
                    region_text_color = region["text_color"]
                    polygon_coords = np.array(region["polygon"].coords, dtype=np.int32)

                    text_size, _ = cv2.getTextSize(
                        region_label, cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.7, thickness=line_thickness)

                    y1 = m1*float(bbox_center[0]) + c1
                    y2 = m2*float(bbox_center[0]) + c2
                    print("line 1" , y1)
                    print("line 2 " , y2)

                    if  -10<bbox_center[1] - ((m1*bbox_center[0]) + c1)  <1 and bbox_center[1] - ((m2*bbox_center[0]) + c2)<-30:
                    
                        color_change1 = 100
                        counter1 = 1
                        print("first if")
                        print(counter1,counter2)
                        
                    if color_change1 >0:
                        cv2.line(frame, data["lines"][0][0],data["lines"][0][1], color=(0, 0, 255), thickness=region_thickness)
                        print(color_change1)
                        color_change1 = color_change1 - 1
                    
                        cv2.rectangle(
                            frame,
                            (data["lines"][0][0][0]- 5, data["lines"][0][0][1] - text_size[1]  - 5),
                            (data["lines"][0][0][0] + text_size[0] + 5, data["lines"][0][0][1] + 5),
                            region_color,
                            -1)
                        cv2.putText(frame, str(counter1), data["lines"][0][0], cv2.FONT_HERSHEY_SIMPLEX, 0.7, region_text_color, line_thickness)

                    if -10< bbox_center[1] - ((m2*bbox_center[0]) + c2)  <1 and (bbox_center[1] - ((m1*bbox_center[0]) + c1)) <-30:
                    
                        color_change2 = 100
                        counter2 = 1
                        
                    if color_change2 >0:
                        print(color_change2)
                        cv2.line(frame, data["lines"][1][0],data["lines"][1][1], color=(0, 0, 255), thickness=region_thickness)
                        color_change2 = color_change2 - 1
                        cv2.rectangle(
                            frame,
                            (data["lines"][1][0][0] - 5, data["lines"][1][0][1] - text_size[1] - 5),
                            (data["lines"][1][0][0] + text_size[0] + 5, data["lines"][1][0][1] + 5),
                            region_color,
                            -1)
                        cv2.putText(frame, str(counter2), data["lines"][1][0], cv2.FONT_HERSHEY_SIMPLEX, 0.7, region_text_color, line_thickness)

        end = datetime.datetime.now()
    # show the time it took to process 1 frame
        
        
        if view_img:
            if vid_frame_count == 1:
                cv2.namedWindow("Ultralytics YOLOv8 Region Counter Movable")
                cv2.setMouseCallback("Ultralytics YOLOv8 Region Counter Movable", mouse_callback)
            total = (end - start).total_seconds()
            # calculate the frame per second and draw it on the frame
            fps = f"FPS: {1 / total:.2f}"
            cv2.putText(frame, fps, (900, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 8)
            cv2.imshow("Ultralytics YOLOv8 Region Counter Movable", frame)

        if save_img:
            
            video_writer.write(frame)

        for region in counting_regions:  # Reinitialize count for each region
            region["counts"] = 0

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    del vid_frame_count
    video_writer.release()
    videocapture.release()
    cv2.destroyAllWindows()


def parse_opt():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, default="yolov8n.pt", help="initial weights path")
    parser.add_argument("--device", default="", help="cuda device, i.e. 0 or 0,1,2,3 or cpu")
    parser.add_argument("--source", type=str,  default="data/input.mp4")
    parser.add_argument("--view-img", default=True, help="show results")
    parser.add_argument("--save-img", default=True, help="save results")
    parser.add_argument("--exist-ok", default=False, help="existing project/name ok, do not increment")
    parser.add_argument("--classes", nargs="+", type=int, help="filter by class: --classes 0, or --classes 0 2 3")
    parser.add_argument("--line-thickness", type=int, default=2, help="bounding box thickness")
    parser.add_argument("--track-thickness", type=int, default=2, help="Tracking line thickness")
    parser.add_argument("--region-thickness", type=int, default=4, help="Region thickness")

    return parser.parse_args()


def main(opt):
    """Main function."""
    run(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)