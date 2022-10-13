from dataclasses import dataclass

@dataclass
class ImageData:
    image_path: str = ""
    user_id: str = ""
    pitch: float = 0.0
    yaw: float = 0.0
