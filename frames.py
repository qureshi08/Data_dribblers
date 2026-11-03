"""Reads chosen frames of a video in one pass.

Seeking to each frame separately (cap.set per frame) makes OpenCV decode from the nearest keyframe every time; deep in
a long upload (a two hour broadcast, analysis from minute 7) that was over 10 s a frame. Here it seeks once to the first
wanted frame and then reads straight through, decoding only what is needed.
"""
import cv2


def frames_at(video, wanted):
    """yields (frame index, image) for each index in `wanted` (any order in, ascending out)"""
    want = sorted(set(int(i) for i in wanted))
    if not want:
        return
    cap = cv2.VideoCapture(str(video))
    cap.set(cv2.CAP_PROP_POS_FRAMES, want[0])
    pos, k = want[0], 0
    while k < len(want):
        if pos < want[k]:
            if not cap.grab():
                break
            pos += 1
            continue
        ok, img = cap.read()
        pos += 1
        if not ok:
            break
        yield want[k], img
        k += 1
    cap.release()
