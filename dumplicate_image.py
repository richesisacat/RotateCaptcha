import os

import cv2
from skimage.metrics import structural_similarity as ssim


def load_images_from_folder(folder):
    images = []
    filenames = []
    for filename in os.listdir(folder):
        if filename.endswith(".png"):
            img = cv2.imread(os.path.join(folder, filename))
            if img is not None:
                images.append(img)
                filenames.append(filename)
    return images, filenames


def rotate_image(image, angle):
    (h, w) = image.shape[:2]
    center = (w / 2, h / 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h))
    return rotated


def compare_images(imageA, imageB):
    imageA = cv2.resize(imageA, (100, 100))
    imageB = cv2.resize(imageB, (100, 100))
    imageA = cv2.cvtColor(imageA, cv2.COLOR_BGR2GRAY)
    imageB = cv2.cvtColor(imageB, cv2.COLOR_BGR2GRAY)
    s, _ = ssim(imageA, imageB, full=True)
    return s


def find_all_duplicates(folder):
    images, filenames = load_images_from_folder(folder)
    visited = set()
    all_duplicates = []

    for i in range(len(images)):
        if filenames[i] in visited:
            continue

        current_group = [filenames[i]]
        visited.add(filenames[i])

        for j in range(i + 1, len(images)):
            if filenames[j] in visited:
                continue

            is_duplicate = False
            for angle in range(0, 360, 15):
                rotated = rotate_image(images[j], angle)
                similarity = compare_images(images[i], rotated)
                if similarity > 0.95:
                    is_duplicate = True
                    break

            if is_duplicate:
                current_group.append(filenames[j])
                visited.add(filenames[j])

        if len(current_group) > 1:
            all_duplicates.append(current_group)

    return all_duplicates


folder_path = 'D:\\work\\py\\RotateCaptcha\\check'  # Change this to your folder path
duplicates = find_all_duplicates(folder_path)

print("Duplicate Image Groups:")
for group in duplicates:
    print("Group:", ", ".join(group))
