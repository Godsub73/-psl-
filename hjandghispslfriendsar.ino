#include <Wire.h>
#include <Adafruit_PWMServoDriver.h>

// Adafruit PWM Servo Driver 객체 초기화
Adafruit_PWMServoDriver pwm = Adafruit_PWMServoDriver();

// 서보 펄스 폭 설정
#define SERVOMIN 120    // 최소값 (구부리기)
#define SERVOMAX 550    // 최대값 (펴기)
#define SERVONEUTRAL 307 // 중립값

// 핀 정의 (서보 연결 위치)
#define THUMB 0
#define INDEX 2
#define MIDDLE 4
#define LITTLE 6

void setup() {
  Serial.begin(9600);  // Serial 통신 시작
  pwm.begin();
  pwm.setPWMFreq(50);  // 50Hz로 설정
  delay(10);

  neutral();  // 모든 서보를 중립 위치로 이동
  Serial.println("Ready for commands...");
}

// 모든 서보를 중립 위치로 이동
void neutral() {
  pwm.setPWM(THUMB, 0, SERVONEUTRAL);    // 엄지 중립
  pwm.setPWM(INDEX, 0, SERVONEUTRAL);    // 검지 중립
  pwm.setPWM(MIDDLE, 0, SERVONEUTRAL);   // 중지 중립
  pwm.setPWM(LITTLE, 0, SERVONEUTRAL);   // 새끼 중립
}

// 서보를 최대값으로 이동 (펴기)
void openHand() {
  pwm.setPWM(THUMB, 0, SERVOMAX);        // 엄지 펴기
  pwm.setPWM(INDEX, 0, SERVOMIN);        // 검지 펴기 (반전)
  pwm.setPWM(MIDDLE, 0, SERVOMAX);       // 중지 펴기
  pwm.setPWM(LITTLE, 0, SERVOMIN);       // 새끼 펴기 (반전)
}

// 서보를 최소값으로 이동 (구부리기)
void closeHand() {
  pwm.setPWM(THUMB, 0, SERVOMIN);        // 엄지 구부리기
  pwm.setPWM(INDEX, 0, SERVOMAX);        // 검지 구부리기 (반전)
  pwm.setPWM(MIDDLE, 0, SERVOMIN);       // 중지 구부리기
  pwm.setPWM(LITTLE, 0, SERVOMAX);       // 새끼 구부리기 (반전)
}

// 검지와 중지는 펴고 나머지는 구부리기
void spreadIndexAndMiddle() {
  pwm.setPWM(THUMB, 0, SERVOMIN);        // 엄지 구부리기
  pwm.setPWM(INDEX, 0, SERVOMIN);        // 검지 펴기 (반전)
  pwm.setPWM(MIDDLE, 0, SERVOMAX);       // 중지 펴기
  pwm.setPWM(LITTLE, 0, SERVOMAX);       // 새끼 구부리기 (반전)
}

// 각각의 손가락 제어 (개별 동작)
void controlFinger(int finger, int position) {
  pwm.setPWM(finger, 0, position);
}

void loop() {
  if (Serial.available() > 0) {
    char command = Serial.read();  // 명령 읽기

    if (command == '0') {
      openHand();  // 손 펴기
    } else if (command == '1') {
      closeHand();  // 손 구부리기
    } else if (command == '2') {
      spreadIndexAndMiddle();  // 검지와 중지는 펴고 나머지는 구부리기
    } else if (command == 'T') {
      controlFinger(THUMB, SERVOMAX);  // 엄지 펴기
    } else if (command == 't') {
      controlFinger(THUMB, SERVOMIN);  // 엄지 구부리기
    } else if (command == 'I') {
      controlFinger(INDEX, SERVOMAX);  // 검지 구부리기 (반전)
    } else if (command == 'i') {
      controlFinger(INDEX, SERVOMIN);  // 검지 펴기 (반전)
    } else if (command == 'M') {
      controlFinger(MIDDLE, SERVOMAX);  // 중지 펴기
    } else if (command == 'm') {
      controlFinger(MIDDLE, SERVOMIN);  // 중지 구부리기
    } else if (command == 'L') {
      controlFinger(LITTLE, SERVOMAX);  // 새끼 구부리기 (반전)
    } else if (command == 'l') {
      controlFinger(LITTLE, SERVOMIN);  // 새끼 펴기 (반전)
    }
    Serial.print("Command received: ");
    Serial.println(command);
  }
}
