
const int PINO_SERVO = 13;
const int ANGULO_MIN = 55;
const int ANGULO_MAX = 125;

void moverServo(int angulo) {
  angulo = constrain(angulo, ANGULO_MIN, ANGULO_MAX);

  // Pulso entre 1000 e 2000 microssegundos
  int pulso = map(angulo, 0, 180, 1000, 2000);

  // PWM de 50 Hz, resolucao de 14 bits
  uint32_t duty = (pulso * 16383UL) / 20000UL;

  ledcWrite(PINO_SERVO, duty);
}

void setup() {
  Serial.begin(115200);
  Serial.setTimeout(30);

  if (!ledcAttach(PINO_SERVO, 50, 14)) {
    Serial.println("Erro ao configurar PWM");
    return;
  }

  moverServo(90);
  delay(500);

  Serial.println("FaceTrack pronto!");
}

void loop() {
  if (Serial.available()) {
    String comando = Serial.readStringUntil('\n');
    comando.trim();

    if (comando.length() > 0 &&
        isDigit(comando.charAt(0))) {
      int angulo = comando.toInt();
      moverServo(angulo);
    }
  }
}
