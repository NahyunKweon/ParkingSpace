import cv2

# 비디오 파일 경로
video_path = 'output.mp4'

# 비디오 캡처 객체 생성
cap = cv2.VideoCapture(video_path)

# 비디오가 열렸는지 확인
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

paused = False  # 일시 정지 상태 변수

while True:
    if not paused:
        # 프레임 읽기
        ret, frame = cap.read()

        # 비디오 끝에 도달하면 종료
        if not ret:
            break

        # 프레임 표시
        cv2.imshow('Video', frame)

    # 키 입력 대기 (1ms)
    key = cv2.waitKey(1)

    # 스페이스바를 눌렀을 때 일시 정지/재생 전환
    if key & 0xFF == ord(' '):
        paused = not paused  # 일시 정지 상태 전환

    # 'q' 키를 눌렀을 때 종료
    if key & 0xFF == ord('q'):
        break

# 자원 해제
cap.release()
cv2.destroyAllWindows()
