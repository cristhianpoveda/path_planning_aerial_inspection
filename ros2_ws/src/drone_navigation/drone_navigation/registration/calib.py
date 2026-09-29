"""Camera intrinsics loader.
"""

import numpy as np
import yaml


def load_calib(path):
    """Accepts flat fx/fy/cx/cy/k1/k2, ORB-SLAM 'Camera.fx' keys, or an
    OpenCV camera_matrix / distortion_coefficients block.

    Returns (K, dist).
    """
    with open(path) as f:
        d = yaml.safe_load(f.read().replace("%YAML:1.0", "").replace("\t", " "))

    if "camera_matrix" in d:
        K = np.array(d["camera_matrix"]["data"], float).reshape(3, 3)
        dist = np.array(d["distortion_coefficients"]["data"], float).ravel()
        return K, dist

    def get(*names, default=None):
        for n in names:
            if n in d:
                return d[n]
        if default is None:
            raise KeyError(f"none of {names} in {path}")
        return default

    fx = float(get("fx", "Camera.fx", "Camera1.fx"))
    fy = float(get("fy", "Camera.fy", "Camera1.fy", default=fx))
    cx = float(get("cx", "Camera.cx", "Camera1.cx"))
    cy = float(get("cy", "Camera.cy", "Camera1.cy"))
    dist = np.array([
        float(get("k1", "Camera.k1", "Camera1.k1", default=0.0)),
        float(get("k2", "Camera.k2", "Camera1.k2", default=0.0)),
        float(get("p1", "Camera.p1", "Camera1.p1", default=0.0)),
        float(get("p2", "Camera.p2", "Camera1.p2", default=0.0)),
        float(get("k3", "Camera.k3", "Camera1.k3", default=0.0))])
    return np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]]), dist
