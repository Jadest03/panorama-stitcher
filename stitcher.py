import cv2 as cv
import numpy as np

def get_homography(img_src, img_dst):
    detector = cv.BRISK_create()
    kp1, des1 = detector.detectAndCompute(img_src, None)
    kp2, des2 = detector.detectAndCompute(img_dst, None)

    if des1 is None or des2 is None:
        raise ValueError("특징점을 충분히 찾을 수 없습니다. 텍스처가 부족하거나 이미지가 너무 큽니다.")

    matcher = cv.DescriptorMatcher_create('BruteForce-Hamming')
    matches = matcher.match(des1, des2)
    
    matches = sorted(matches, key=lambda x: x.distance)
    good_matches = matches[:150] 

    pts_src = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    pts_dst = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    H, mask = cv.findHomography(pts_src, pts_dst, cv.RANSAC, 5.0)
    return H

def resize_img(img, max_width=800):
    h, w = img.shape[:2]
    if w > max_width:
        ratio = max_width / w
        return cv.resize(img, (max_width, int(h * ratio)))
    return img

def extract_keyframes(video_path, num_keyframes=5, max_width=800):
    """
    동영상에서 일정한 간격으로 핵심 프레임(Keyframe)을 추출합니다.
    """
    cap = cv.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"비디오 파일을 열 수 없습니다: {video_path}")
        return []

    total_frames = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
    print(f"총 프레임 수: {total_frames}")

    # 영상 처음부터 끝까지 일정한 간격으로 프레임 인덱스 계산
    interval = total_frames // num_keyframes
    frames = []

    for i in range(num_keyframes):
        # 캡처할 프레임 위치로 이동
        frame_idx = i * interval
        cap.set(cv.CAP_PROP_POS_FRAMES, frame_idx)
        
        ret, frame = cap.read()
        if ret:
            # 해상도를 줄여서 리스트에 저장
            frames.append(resize_img(frame, max_width))
            print(f"Keyframe {i+1}/{num_keyframes} 추출 완료 (Frame Index: {frame_idx})")
        else:
            break

    cap.release()
    return frames

def main():
    # 1. 비디오 경로 설정 (테스트용 비디오를 여기에 넣으세요)
    video_path = './IMG_0676.MOV' 
    
    # 2. 비디오에서 5장의 키프레임 추출 (너무 많으면 원근 왜곡이 심해집니다)
    print("비디오 프레임 추출을 시작합니다...")
    images = extract_keyframes(video_path, num_keyframes=20)
    
    num_images = len(images)
    if num_images < 2:
        print("프레임을 충분히 추출하지 못했습니다.")
        return

    print("프레임 스티칭 연산을 시작합니다. 잠시만 기다려주세요...")
    base_idx = num_images // 2  

    # 3. 누적 호모그래피(Chained Homography) 계산
    H_to_base = [np.eye(3, dtype=np.float32) for _ in range(num_images)]

    for i in range(base_idx + 1, num_images):
        H_adj = get_homography(images[i], images[i - 1])
        H_to_base[i] = H_to_base[i - 1] @ H_adj

    for i in range(base_idx - 1, -1, -1):
        H_adj = get_homography(images[i], images[i + 1])
        H_to_base[i] = H_to_base[i + 1] @ H_adj

    # 4. 동적 캔버스 크기 계산
    min_x, min_y = np.inf, np.inf
    max_x, max_y = -np.inf, -np.inf

    for i in range(num_images):
        h, w = images[i].shape[:2]
        corners = np.float32([[0, 0], [0, h], [w, h], [w, 0]]).reshape(-1, 1, 2)
        warped_corners = cv.perspectiveTransform(corners, H_to_base[i])
        
        min_x = min(min_x, np.min(warped_corners[:, 0, 0]))
        min_y = min(min_y, np.min(warped_corners[:, 0, 1]))
        max_x = max(max_x, np.max(warped_corners[:, 0, 0]))
        max_y = max(max_y, np.max(warped_corners[:, 0, 1]))

    canvas_w = int(np.ceil(max_x - min_x))
    canvas_h = int(np.ceil(max_y - min_y))

    T = np.array([[1, 0, -min_x],
                  [0, 1, -min_y],
                  [0, 0, 1]], dtype=np.float32)

    stitched_sum = np.zeros((canvas_h, canvas_w, 3), dtype=np.float32)
    weight_map = np.zeros((canvas_h, canvas_w, 3), dtype=np.float32)

    # 5. 이미지 Warping 및 Blending
    for i in range(num_images):
        H_final = T @ H_to_base[i]
        
        warped_img = cv.warpPerspective(images[i], H_final, (canvas_w, canvas_h))
        stitched_sum += warped_img.astype(np.float32)

        mask = np.ones_like(images[i], dtype=np.float32)
        warped_mask = cv.warpPerspective(mask, H_final, (canvas_w, canvas_h))
        weight_map += warped_mask

    weight_map[weight_map == 0] = 1 
    panorama = (stitched_sum / weight_map).astype(np.uint8)

    print("스티칭 완료! 이미지를 띄웁니다.")
    cv.imshow(f'Video Panorama Stitched ({num_images} Keyframes)', panorama)
    cv.imwrite('video_panorama_final.jpg', panorama)
    cv.waitKey(0)
    cv.destroyAllWindows()

if __name__ == '__main__':
    main()