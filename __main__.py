import glob
import os
import pickle
import re
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import tesserocr
from dotenv import load_dotenv
from PIL import Image
from scenedetect import ContentDetector, detect, split_video_ffmpeg
from scenedetect.frame_timecode import FrameTimecode
from tqdm import tqdm

# TODO: incorporate this:
MIN_SECONDS_PER_SCENE = 60


def main():
    load_dotenv()
    video_path = os.getenv("VIDEO_PATH")
    out_path = os.path.join(Path(video_path).parent, Path(video_path).stem + "_out.pkl")
    if not os.path.isfile(out_path):
        scene_list = compute_scene_list(
            video_path,
            out_path,
        )
    else:
        with open(out_path, "rb") as file:
            scene_list = pickle.load(file)
    filtered_scene_list, dates = filter_scenes(scene_list)
    print_scenes(filtered_scene_list, dates)
    split_video_ffmpeg(
        video_path,
        filtered_scene_list,
        output_dir=Path(video_path).parent,
        output_file_template="$VIDEO_NAME $SCENE_NUMBER.mp4",
        arg_override="-c copy",
        show_progress=True,
    )
    for f in sorted(glob.glob(f"{video_path[:-4]} *")):
        f_split = f.split(" ")
        scene_idx = int(f_split[1][:-4]) - 1
        if scene_idx > 0:
            start_date = dates[scene_idx]
        else:
            start_date = datetime.strptime(Path(video_path).stem[:8], "%y.%m.%d")
        if scene_idx < len(filtered_scene_list) - 1:
            end_date = dates[scene_idx + 1]
        else:
            end_date = datetime.strptime(Path(video_path).stem[9:17], "%y.%m.%d")
        name = start_date.strftime("%Y-%m-%d") + "_-_" + end_date.strftime("%Y-%m-%d")
        out_filename = os.path.join(Path(f).parent, f"{name}.mp4")
        os.rename(f, out_filename)


def compute_scene_list(video_path, out_path):
    pickle_file_path = Path(video_path).with_suffix(".pkl")
    if os.path.isfile(pickle_file_path):
        with open(pickle_file_path, "rb") as file:
            scene_list = pickle.load(file)
    else:
        scene_list = detect(video_path, ContentDetector(), show_progress=True)
        with open(pickle_file_path, "wb") as file:
            pickle.dump(scene_list, file)

    scene_list = ocr_video(video_path, scene_list)
    with open(out_path, "wb") as file:
        pickle.dump(scene_list, file)
    return scene_list


def ocr_video(video_path, scene_list):
    api = tesserocr.PyTessBaseAPI(
        path="/usr/share/tessdata/",
        variables={"tessedit_char_whitelist": "0123456789."},
        psm=tesserocr.PSM.SINGLE_BLOCK,
    )

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    roi_x, roi_y, roi_w, roi_h = 60, 380, 280, 160

    frame_idx = 0
    scene_idx = 0
    rois = []
    start_frame = scene_list[scene_idx][0].get_frames()
    last_timecode = datetime.strptime(Path(video_path).stem[:8], "%y.%m.%d")
    end_timecode = datetime.strptime(Path(video_path).stem[9:17], "%y.%m.%d")
    with tqdm(total=total_frames, desc="Processing Video", unit="frame") as pbar:
        while cap.isOpened():
            ret, frame = cap.read()
            frame_idx += 1
            pbar.update(1)
            if not ret:
                break
            if frame_idx < start_frame:
                continue

            roi = frame[roi_y : roi_y + roi_h, roi_x : roi_x + roi_w]
            rois.append(roi.astype(np.float32))

            if len(rois) >= fps * 3:  # 3 seconds
                mean_roi = np.mean(rois, axis=0).astype(np.uint8)
                rois = []
                gray_roi = cv2.cvtColor(mean_roi, cv2.COLOR_BGR2GRAY)
                new_size = (int(1.2 * gray_roi.shape[1]), int(1.2 * gray_roi.shape[0]))
                gray_roi = cv2.resize(gray_roi, new_size)
                inv_roi = 255 - gray_roi
                # cv2.imwrite("gray_roi.png", inv_roi)
                sobel_x = cv2.Sobel(inv_roi, cv2.CV_64F, 1, 0, ksize=3)
                sobel_x = cv2.convertScaleAbs(sobel_x * 2)
                sobel_y = cv2.Sobel(inv_roi, cv2.CV_64F, 0, 1, ksize=3)
                sobel_y = cv2.convertScaleAbs(sobel_y * 2)
                sobel = cv2.addWeighted(sobel_x, 0.5, sobel_y, 0.5, 0)
                _, thresh = cv2.threshold(sobel, 127, 255, cv2.THRESH_BINARY)

                contour, _ = cv2.findContours(
                    thresh, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE
                )
                for cnt in contour:
                    cv2.drawContours(thresh, [cnt], 0, 255, -1)
                inv_mask = cv2.bitwise_not(thresh)

                # cv2.imwrite("sobel_x.png", inv_mask)

                # extract text from the ROI
                api.SetImage(Image.fromarray(inv_roi))
                text1 = api.GetUTF8Text()
                api.SetImage(Image.fromarray(inv_mask))
                text2 = api.GetUTF8Text()

                for text in [text1, text2]:
                    if not text:
                        continue
                    timecode = parse_timecode(text)
                    if (
                        timecode
                        and timecode > last_timecode
                        and timecode <= end_timecode
                    ):
                        start, end = get_timecodes(scene_list, scene_idx)
                        print(f"Scene {scene_idx}: {text} - Start {start}, End {end}")
                        print(f"Timecode {timecode}")
                        scene_list[scene_idx] += (timecode,)
                        last_timecode = timecode
                        break

                scene_idx += 1
                start_frame = scene_list[scene_idx][0].get_frames()

    cap.release()
    return scene_list


def filter_scenes(scene_list):
    filtered_scene_list = []
    dates = []
    for scene in scene_list:
        if len(scene) < 3:
            continue
        else:
            if filtered_scene_list:
                tmp = list(filtered_scene_list[-1])
                tmp[1] = FrameTimecode(
                    scene[0].get_timecode(), scene[0].get_framerate()
                )
                filtered_scene_list[-1] = tuple(tmp)

                tmp = list(scene)
                dates.append(tmp.pop())
                filtered_scene_list.append(tuple(tmp))
            else:  # empty list
                tmp = list(scene)
                tmp[0] = FrameTimecode("00:00:00.000", scene[0].get_framerate())
                dates.append(tmp.pop())
                filtered_scene_list.append(tuple(tmp))
    end_t = scene_list[-1][1].get_timecode()
    tmp = list(filtered_scene_list[-1])
    tmp[1] = FrameTimecode(end_t, scene_list[-1][1].get_framerate())
    filtered_scene_list[-1] = tuple(tmp)
    return filtered_scene_list, dates


def print_scenes(scene_list, dates):
    for i, scene in enumerate(scene_list):
        print(
            "Scene %2d: %s - Start %s / Frame %d, End %s / Frame %d"
            % (
                i + 1,
                dates[i],
                scene[0].get_timecode(),
                scene[0].get_frames(),
                scene[1].get_timecode(),
                scene[1].get_frames(),
            )
        )


def get_scene_len(scene_list, scene_idx):
    target_frame_cnt = (
        scene_list[scene_idx][1].get_frames() - scene_list[scene_idx][0].get_frames()
    )
    return target_frame_cnt


def get_timecodes(scene_list, scene_idx):
    return (
        scene_list[scene_idx][0].get_timecode(),
        scene_list[scene_idx][1].get_timecode(),
    )


def parse_timecode(text):
    m = re.search(r"\d\d?\.\d\d?\.\d\d\d\d", text)
    try:
        date_str = m.group()
        return datetime.strptime(date_str, "%d.%m.%Y")
    except:
        return None


if __name__ == "__main__":
    main()
