import cv2
import numpy as np
import matplotlib.pyplot as plt
from utils import classify_cloud_shape, simplify_contour, find_contours

SCORE_TH = 0.3 

def main():
    # Load the cloud image
    image = cv2.imread('images/clouds_1.jpg')
    output_image = image.copy()
    transparent_image = np.zeros((image.shape[0], image.shape[1], 4), dtype=np.uint8)

    # Find clouds contours and filter small ones
    contours = find_contours(image)

    # Process each detected cloud contour
    for i, contour in enumerate(contours):
        simplified_contour = simplify_contour(contour)
        mask = np.zeros_like(image)
        cv2.drawContours(mask, [simplified_contour], -1, (255, 255, 255), thickness=2)
        mask_rgb = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)

               
        # Visualize mask_rgb
        plt.imshow(mask_rgb)
        plt.axis('off')
        plt.savefig(f'mask_rgb_{i+1}.png', bbox_inches='tight', pad_inches=0)
        plt.close()


        # Classify and label cloud shape
        label, score = classify_cloud_shape(mask_rgb)
        print(f"Cloud Shape {i+1}: {label} ({score:.2f})")

        # Draw simplified contour and label if > SCORE_TH
        cv2.drawContours(output_image, [simplified_contour], -1, (255, 255, 255), thickness=2)
        if score > SCORE_TH:
            cv2.putText(output_image, f"{label} ({score:.2f})",
                        (simplified_contour[0][0][0], simplified_contour[0][0][1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(transparent_image, f"{label} ({score:.2f})", 
                        (simplified_contour[0][0][0], simplified_contour[0][0][1] - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255, 255), 1)
        cv2.drawContours(transparent_image, [simplified_contour], -1, (255, 255, 255, 255), 1)

    # Save results
    cv2.imwrite('output/cloud_image_with_contours.jpg', output_image)
    cv2.imwrite('output/cloud_outlines_transparent.png', transparent_image)

if __name__ == "__main__":
    main()
