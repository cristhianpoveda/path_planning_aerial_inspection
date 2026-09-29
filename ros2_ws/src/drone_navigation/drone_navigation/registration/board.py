"""AprilGrid board model, detection and PnP.

Board: 6x6 tag36h11, tag size 0.088 m, gap 0.0264 m,
IDs 0..35 left-to-right then bottom-to-top.
"""

import cv2
import numpy as np

TAG_ROWS = 6
TAG_COLS = 6
TAG_SIZE = 0.088
TAG_GAP = 0.0264
PITCH = TAG_SIZE + TAG_GAP          # 0.1144
N_TAGS = TAG_ROWS * TAG_COLS

_CORNER_UV = ((1, 0), (0, 0), (0, 1), (1, 1))

# Top edge of the top row of corner squares, in the kalibr frame.
_Y_TOP = (TAG_ROWS - 1) * PITCH + TAG_SIZE + TAG_GAP      # 0.6864

# Pose of the user frame expressed in the kalibr frame, and its inverse.
T_KALIBR_USER = np.array([[1.0, 0.0, 0.0, -TAG_GAP],
                          [0.0, -1.0, 0.0, _Y_TOP],
                          [0.0, 0.0, -1.0, 0.0],
                          [0.0, 0.0, 0.0, 1.0]])
T_USER_KALIBR = np.array([[1.0, 0.0, 0.0, TAG_GAP],
                          [0.0, -1.0, 0.0, _Y_TOP],
                          [0.0, 0.0, -1.0, 0.0],
                          [0.0, 0.0, 0.0, 1.0]])

CORNER_INSET = 0.0068


def object_points(tag_id, inset=None):
    """Corners of one tag in the kalibr board frame, in cv2.aruco order."""
    ins = CORNER_INSET if inset is None else inset
    e = TAG_SIZE - 2 * ins
    r, c = divmod(int(tag_id), TAG_COLS)
    x0, y0 = c * PITCH + ins, r * PITCH + ins
    return np.array([[x0 + u * e, y0 + v * e, 0.0]
                     for u, v in _CORNER_UV], dtype=np.float64)


def make_detector():
    """cv2.aruco, not pupil_apriltags.
    """
    d = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
    p = cv2.aruco.DetectorParameters()
    p.adaptiveThreshWinSizeMin = 3
    p.adaptiveThreshWinSizeMax = 53
    p.adaptiveThreshWinSizeStep = 4
    p.minMarkerPerimeterRate = 0.01
    p.maxErroneousBitsInBorderRate = 0.4
    p.errorCorrectionRate = 0.8
    p.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    return cv2.aruco.ArucoDetector(d, p)


def _to_T(rvec, tvec):
    T = np.eye(4)
    T[:3, :3] = cv2.Rodrigues(rvec)[0]
    T[:3, 3] = tvec.reshape(3)
    return T


def detect_board(gray, detector, K, dist, min_tags=6, inset=None):
    """Return (T_cam_kalibr, n_tags, rms_px) or None.
    """
    corners, ids, _ = detector.detectMarkers(gray)
    if ids is None:
        return None
    ids = ids.flatten()
    keep = [i for i, t in enumerate(ids) if 0 <= t < N_TAGS]
    if len(keep) < min_tags:
        return None

    obj = np.vstack([object_points(ids[i], inset) for i in keep])
    img = np.vstack([corners[i][0] for i in keep]).astype(np.float64)

    ok, rvec, tvec = cv2.solvePnP(obj, img, K, dist, flags=cv2.SOLVEPNP_IPPE)
    if not ok:
        return None
    rvec, tvec = cv2.solvePnPRefineLM(obj, img, K, dist, rvec, tvec)

    proj, _ = cv2.projectPoints(obj, rvec, tvec, K, dist)
    rms = float(np.sqrt(np.mean(np.sum(
        (proj.reshape(-1, 2) - img) ** 2, axis=1))))
    return _to_T(rvec, tvec), len(keep), rms


def verify_corner_order(gray, detector):
    """Self-check for _CORNER_UV.  Fits a homography from tag centres, then
    scores all four cyclic corner orderings against the detected corners.
    """
    corners, ids, _ = detector.detectMarkers(gray)
    if ids is None:
        return None
    ids = ids.flatten()
    B, I = [], []
    for k, t in enumerate(ids):
        if not 0 <= t < N_TAGS:
            continue
        r, c = divmod(int(t), TAG_COLS)
        B.append([c * PITCH + TAG_SIZE / 2, r * PITCH + TAG_SIZE / 2])
        I.append(corners[k][0].mean(axis=0))
    H, _ = cv2.findHomography(np.array(B, np.float32),
                              np.array(I, np.float32), cv2.RANSAC, 3.0)

    def proj(pts):
        P = np.hstack([pts, np.ones((len(pts), 1))]).T
        q = H @ P
        return (q[:2] / q[2]).T

    base = [(1, 0), (0, 0), (0, 1), (1, 1)]
    out = {}
    for shift in range(4):
        off = base[shift:] + base[:shift]
        err = []
        for k, t in enumerate(ids):
            if not 0 <= t < N_TAGS:
                continue
            r, c = divmod(int(t), TAG_COLS)
            x0, y0 = c * PITCH, r * PITCH
            o = np.array([[x0 + u * TAG_SIZE, y0 + v * TAG_SIZE]
                          for u, v in off], np.float32)
            err.append(np.linalg.norm(proj(o) - corners[k][0], axis=1))
        out[tuple(off)] = float(np.mean(err))
    return out


def fit_corner_inset(gray, detector, K, dist):
    """Measure CORNER_INSET on one frame by minimising reprojection rms.
    """
    from scipy.optimize import least_squares
    corners, ids, _ = detector.detectMarkers(gray)
    if ids is None:
        return None
    ids = ids.flatten()
    keep = [i for i, t in enumerate(ids) if 0 <= t < N_TAGS]
    img = np.vstack([corners[i][0] for i in keep]).astype(np.float64)

    def rms(x):
        obj = np.vstack([object_points(ids[i], x[0]) for i in keep])
        ok, rv, tv = cv2.solvePnP(obj, img, K, dist, flags=cv2.SOLVEPNP_IPPE)
        rv, tv = cv2.solvePnPRefineLM(obj, img, K, dist, rv, tv)
        pr, _ = cv2.projectPoints(obj, rv, tv, K, dist)
        return np.sqrt(np.mean(np.sum(
            (pr.reshape(-1, 2) - img) ** 2, axis=1)))

    s = least_squares(lambda x: rms(x), [0.005],
                      bounds=([0.0], [TAG_SIZE / 4]))
    return float(s.x[0]), float(s.fun[0]), float(rms([0.0]))
