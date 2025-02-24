import cv2
import tensorflow as tf
import numpy as np
import serial  # 아두이노와의 Serial 통신을 위한 라이브러리
import time
qq
# 아두이노 연결 설정
# 아두이노와 통신을 하기 위해 COM 포트를 지정하고 초기화를 기다림.
# 여기서는 COM5로 설정했고, 보드 속도를 9600bps로 설정.
# 아두이노 연결 확인 후, 약 2초 정도 초기화 시간을 기다림.
# 오류: 초기화 대기를 넣지 않아 아두이노가 데이터를 받지 못함. 
arduino_port = "COM5"  # 아두이노가 연결된 COM 포트
baud_rate = 9600       # 아두이노 시리얼 통신 속도
arduino = serial.Serial(arduino_port, baud_rate)
time.sleep(2)          # 아두이노 초기화 대기

# 모델 로드
# TensorFlow로 변환된 모델을 불러오는 부분.
# 모델 경로는 절대 경로를 사용했는데, 이 부분에서 로드 오류가 났던 적이 있었음.
# 오류: 모델 경로 오타로 파일을 로드하지 못함. 
model_path = "C:/Users/hj070/Downloads/converted_savedmodel/model.savedmodel"
model = tf.saved_model.load(model_path)

# 모델 서명을 가져옴
# TensorFlow 모델은 'serving_default'라는 서명을 통해 입력과 출력을 정의함.
# 모델 서명을 정확히 확인하지 않으면 추론 단계에서 오류 발생 가능.
# 오류: 서명 이름 오타로 추론이 작동하지 않
infer = model.signatures['serving_default']

# 웹캠 초기화
# 웹캠을 통해 실시간 영상을 읽어옴. 기본 웹캠 장치(0)를 사용.
# 초기 테스트에서는 웹캠이 제대로 연결되지 않아서 오류가 났었는데,
# 카메라 권한 문제를 해결하고 나서 정상 동작함.
# 오류: 카메라 권한 미설정으로 웹캠이 열리지 않음. 
cap = cv2.VideoCapture(0)

# 웹캠이 열리지 않을 경우 종료
# 장치 연결 문제를 빠르게 확인할 수 있도록 예외 처리.
if not cap.isOpened():
    print("웹캠을 열 수 없습니다.")
    exit()

# 메인 루프
# 실시간 영상 처리 및 예측 결과를 아두이노로 전송.
while True:
    # 웹캠에서 프레임 캡처
    # 초기에는 캡처가 불안정하게 작동해서 코드가 멈추는 경우가 있었음.
    # 이를 확인하기 위해 프레임 캡처 실패 시 로그를 출력하도록 추가.
    # 오류: 캡처 실패 시 처리 루틴 누락. 
    ret, frame = cap.read()

    # 프레임이 제대로 캡처되지 않으면 종료
    if not ret:
        print("프레임을 가져올 수 없습니다.")
        break

    # 웹캠 영상 반전
    # 수평 반전 처리를 넣은 이유는 웹캠이 거울처럼 보이게 하기 위해서였음.
    frame = cv2.flip(frame, 1)

    # 이미지 전처리
    # 모델 입력에 맞게 이미지를 리사이즈하고 정규화하는 과정.
    # 초반에 정규화를 잊어서 추론 결과가 전혀 엉뚱하게 나왔던 적이 있었음.
    # 오류: 정규화 누락으로 잘못된 예측 결과 발생.
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # BGR -> RGB
    image = cv2.resize(image, (224, 224))          # 모델 입력 크기 (224, 224)
    image = np.expand_dims(image, axis=0)          # 배치 차원 추가
    image = image / 255.0                          # 정규화 (0~1)

    # 모델 추론
    # 입력 이미지를 Tensor로 변환하여 모델에 전달.
    # 추론 결과를 확인할 때 'sequential_7' 키를 사용했는데, 모델 구조에 따라 달라질 수 있으니 주의해야 함.
    # 오류: 키 이름 변경 시 누락된 코드 수정 필요. 
    input_tensor = tf.convert_to_tensor(image, dtype=tf.float32)
    prediction = infer(input_tensor)

    # 예측된 클래스 가져오기
    # 모델 출력 중 가장 높은 확률을 가진 클래스를 선택.
    # 여기서 np.argmax를 사용하는데, 초기에 axis를 잘못 지정해서 결과가 틀리게 나온 적이 있었음.
    # 오류: np.argmax axis 설정 오류. 
    output = prediction['sequential_7'].numpy()
    predicted_class = np.argmax(output, axis=1)[0]

    # 예측 결과를 아두이노로 전송
    # 클래스 값을 문자열로 변환해 아두이노로 보냄.
    # 초기에는 '\n'을 빼먹어서 아두이노가 데이터를 제대로 처리하지 못했음.
    # 오류: 줄바꿈 문자 누락으로 아두이노에서 데이터 해석 실패.
    arduino.write(f"{predicted_class}\n".encode('utf-8'))

    # 예측 결과를 화면에 출력
    # OpenCV의 putText를 이용해 프레임에 예측 결과를 실시간으로 표시.
    cv2.putText(frame, f"Prediction: {predicted_class}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # 실시간 영상 출력
    # OpenCV의 imshow를 통해 실시간으로 처리 결과를 확인.
    # 영상이 끊기지 않고 자연스럽게 출력되도록 성능 튜닝이 필요했음.
    cv2.imshow('Webcam Feed', frame)

    # 'q' 키를 누르면 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 리소스 해제
# 모든 자원을 정리하고 프로그램을 종료.
# 아두이노 포트와 웹캠 장치를 제대로 해제하지 않으면 다음 실행 시 충돌이 발생할 수 있음.
cap.release()
cv2.destroyAllWindows()
arduino.close()

# 크레딧
