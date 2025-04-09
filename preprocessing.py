def bilateral_denoise(image):
    return cv2.bilateralFilter(image.astype(np.float32), d=9, sigmaColor=75, sigmaSpace=75)

def apply_clahe(image):
    norm = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(norm)

def sobel_edges(image):
    grad_x = cv2.Sobel(image, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(image, cv2.CV_32F, 0, 1, ksize=3)
    magnitude = cv2.magnitude(grad_x, grad_y)
    return magnitude

def downsample(image, target_shape=(64, 64)):
    return resize(image, target_shape, mode='reflect', preserve_range=True, anti_aliasing=True)
