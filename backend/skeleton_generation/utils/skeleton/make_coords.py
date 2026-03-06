from pathlib import Path

import cv2


TARGET_HEIGHT = 200
IMAGES_DIRECTORY = Path("M-to-PY_Skel/original")
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def extract_contours(image, target_height=TARGET_HEIGHT):
    """Resize input image and extract external contours from Canny edges."""
    if image is None:
        raise ValueError("Input image cannot be None.")
    if image.ndim < 2:
        raise ValueError("Input image must be at least 2D.")
    if target_height <= 0:
        raise ValueError("target_height must be positive.")

    src_h, src_w = image.shape[:2]
    target_width = max(1, int(target_height * (src_w / src_h)))

    resized = cv2.resize(image, (target_width, target_height), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, threshold1=30, threshold2=100)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours, edges.shape[0]


def write_contours_txt(contours, height, output_path):
    """Write contour coordinates in the legacy y-flipped format."""
    with output_path.open("w", encoding="utf-8") as handle:
        for contour in contours:
            points = contour.reshape(-1, 2)
            for x, y in points:
                handle.write(f"{float(x):.7e} {float(height - y):.7e}\n")


def process_directory(images_directory=IMAGES_DIRECTORY, target_height=TARGET_HEIGHT):
    if not images_directory.exists():
        raise FileNotFoundError(f"Directory not found: {images_directory}")

    for image_path in sorted(images_directory.iterdir()):
        if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue

        image = cv2.imread(str(image_path))
        if image is None:
            # Keep batch processing resilient if one file is unreadable.
            continue

        contours, height = extract_contours(image, target_height=target_height)
        output_path = image_path.with_suffix(".txt")
        write_contours_txt(contours, height, output_path)


if __name__ == "__main__":
    process_directory()
