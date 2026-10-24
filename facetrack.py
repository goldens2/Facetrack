
import cv2
import mediapipe as mp
import serial
import time
from pathlib import Path

# ======================================
# FACETRACK 4.0 - MEDIAPIPE
# ======================================

PORTA = "COM3"
CAMERA = 0
SENTIDO = -1  # Use -1 se o giro ficar invertido

# Movimento
ANGULO_MIN = 55
ANGULO_MAX = 125
ANGULO_INICIAL = 90

ZONA_MORTA = 25
GANHO = 0.06
PASSO_MAXIMO = 5.0
INTERVALO = 0.08
SUAVIZACAO = 0.4

# Busca automatica
TEMPO_BUSCA = 1.5
PASSO_BUSCA = 3.0

# Modelo MediaPipe
MODELO = Path(__file__).with_name(
    "face_landmarker.task"
)

if not MODELO.exists():
    raise FileNotFoundError(
        f"Modelo nao encontrado: {MODELO}"
    )

# ======================================
# COMUNICACAO E CAMARA
# ======================================

esp = serial.Serial(PORTA, 115200, timeout=1)
time.sleep(2)

camera = cv2.VideoCapture(CAMERA)
camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

angulo = float(ANGULO_INICIAL)
ultimo_angulo = ANGULO_INICIAL
ultimo_movimento = 0

centro_suavizado = None
ultimo_rosto = time.monotonic()
direcao_busca = 1

pausado = False
mostrar_hud = True
mostrar_malha = False

fps = 0
inicio = time.monotonic()
ultimo_frame = inicio
ultimo_timestamp = -1


def mover_servo(destino):
    global angulo, ultimo_angulo

    angulo = max(
        ANGULO_MIN,
        min(ANGULO_MAX, destino)
    )

    novo = round(angulo)

    if novo != ultimo_angulo:
        esp.write(f"{novo}\n".encode())
        ultimo_angulo = novo


def desenhar_rosto(imagem, pontos, largura, altura):
    conexoes = (
        mp.tasks.vision
        .FaceLandmarksConnections
        .FACE_LANDMARKS_FACE_OVAL
    )

    # Contorno facial discreto
    for conexao in conexoes:
        a = pontos[conexao.start]
        b = pontos[conexao.end]

        p1 = (
            int(a.x * largura),
            int(a.y * altura)
        )
        p2 = (
            int(b.x * largura),
            int(b.y * altura)
        )

        cv2.line(
            imagem, p1, p2,
            (70, 230, 120), 1
        )

    # Pontos principais do rosto
    for indice in [1, 33, 263, 234, 454]:
        ponto = pontos[indice]

        cv2.circle(
            imagem,
            (int(ponto.x * largura),
             int(ponto.y * altura)),
            2, (0, 255, 255), -1
        )

    # Malha completa opcional
    if mostrar_malha:
        for ponto in pontos:
            cv2.circle(
                imagem,
                (int(ponto.x * largura),
                 int(ponto.y * altura)),
                1, (100, 200, 100), -1
            )


# ======================================
# INICIALIZACAO DO MEDIAPIPE
# ======================================

visao = mp.tasks.vision

opcoes = visao.FaceLandmarkerOptions(
    base_options=mp.tasks.BaseOptions(
        model_asset_path=str(MODELO)
    ),
    running_mode=visao.RunningMode.VIDEO,
    num_faces=1,
    min_face_detection_confidence=0.5,
    min_face_presence_confidence=0.5,
    min_tracking_confidence=0.5
)

print("FaceTrack 4.0 iniciado!")
print("H = ocultar painel")
print("M = mostrar/ocultar malha")
print("P = pausar/retomar")
print("R = recentralizar e pausar")
print("+ / - = ajustar velocidade")
print("Q = sair")

try:
    if not camera.isOpened():
        raise RuntimeError("Webcam nao encontrada")

    with visao.FaceLandmarker.create_from_options(
        opcoes
    ) as detector:

        mover_servo(ANGULO_INICIAL)

        while True:
            ok, imagem = camera.read()

            if not ok:
                break

            agora = time.monotonic()

            duracao = agora - ultimo_frame
            ultimo_frame = agora

            if duracao > 0:
                fps = (
                    fps * 0.85 +
                    (1 / duracao) * 0.15
                )

            altura, largura = imagem.shape[:2]
            centro_tela = largura // 2

            # ==================================
            # DETECCAO COM MEDIAPIPE
            # ==================================

            rgb = cv2.cvtColor(
                imagem, cv2.COLOR_BGR2RGB
            )

            imagem_mp = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb
            )

            timestamp = max(
                ultimo_timestamp + 1,
                int((agora - inicio) * 1000)
            )

            ultimo_timestamp = timestamp

            resultado = detector.detect_for_video(
                imagem_mp, timestamp
            )

            status = "PROCURANDO"
            erro = 0

            # ==================================
            # RASTREAMENTO
            # ==================================

            if resultado.face_landmarks:
                pontos = resultado.face_landmarks[0]
                ultimo_rosto = agora

                # Centro entre as laterais do rosto
                rosto_x = (
                    pontos[234].x +
                    pontos[454].x
                ) / 2 * largura

                # Filtro de suavizacao
                if centro_suavizado is None:
                    centro_suavizado = rosto_x
                else:
                    centro_suavizado = (
                        centro_suavizado *
                        (1 - SUAVIZACAO) +
                        rosto_x * SUAVIZACAO
                    )

                erro = (
                    centro_suavizado -
                    centro_tela
                )

                desenhar_rosto(
                    imagem, pontos,
                    largura, altura
                )

                # Centro do rosto detectado
                cv2.circle(
                    imagem,
                    (int(centro_suavizado),
                     altura // 2),
                    4, (0, 200, 255), -1
                )

                if pausado:
                    status = "PAUSADO"

                elif abs(erro) <= ZONA_MORTA:
                    status = "CENTRO"

                else:
                    status = "SEGUINDO"

                    if (
                        agora - ultimo_movimento
                        >= INTERVALO
                    ):
                        correcao = erro * GANHO

                        correcao = max(
                            -PASSO_MAXIMO,
                            min(
                                PASSO_MAXIMO,
                                correcao
                            )
                        )

                        mover_servo(
                            angulo +
                            correcao * SENTIDO
                        )

                        ultimo_movimento = agora

            # ==================================
            # BUSCA AUTOMATICA
            # ==================================

            else:
                centro_suavizado = None

                tempo_perdido = agora - ultimo_rosto

                if pausado:
                    status = "PAUSADO"

                elif tempo_perdido < TEMPO_BUSCA:
                    status = "ROSTO PERDIDO"

                else:
                    status = "BUSCANDO"

                    if (
                        agora - ultimo_movimento
                        >= INTERVALO
                    ):
                        destino = (
                            angulo +
                            PASSO_BUSCA *
                            direcao_busca
                        )

                        if destino >= ANGULO_MAX:
                            destino = ANGULO_MAX
                            direcao_busca = -1

                        elif destino <= ANGULO_MIN:
                            destino = ANGULO_MIN
                            direcao_busca = 1

                        mover_servo(destino)

                        ultimo_movimento = agora

            # ==================================
            # INTERFACE COMPACTA
            # ==================================

            if mostrar_hud:
                # Barra pequena no topo
                cv2.rectangle(
                    imagem,
                    (0, 0),
                    (largura, 29),
                    (20, 26, 32),
                    -1
                )

                texto = (
                    f"FACETRACK 4.0 | {status} | "
                    f"{ultimo_angulo} deg | "
                    f"{fps:.0f} FPS"
                )

                cv2.putText(
                    imagem,
                    texto,
                    (9, 19),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (130, 255, 185),
                    1,
                    cv2.LINE_AA
                )

                # Pequena marca central
                cv2.line(
                    imagem,
                    (centro_tela, 32),
                    (centro_tela, 53),
                    (255, 220, 70),
                    1
                )

            cv2.imshow(
                "FaceTrack 4.0",
                imagem
            )

            # ==================================
            # CONTROLES
            # ==================================

            tecla = cv2.waitKey(1) & 0xFF

            if tecla == ord("q"):
                break

            elif tecla == ord("p"):
                pausado = not pausado
                ultimo_rosto = time.monotonic()

            elif tecla == ord("r"):
                pausado = True
                mover_servo(ANGULO_INICIAL)
                centro_suavizado = None

            elif tecla == ord("h"):
                mostrar_hud = not mostrar_hud

            elif tecla == ord("m"):
                mostrar_malha = not mostrar_malha

            elif tecla in [ord("+"), ord("=")]:
                GANHO = min(0.20, GANHO + 0.01)
                print(f"Ganho: {GANHO:.2f}")

            elif tecla in [ord("-"), ord("_")]:
                GANHO = max(0.01, GANHO - 0.01)
                print(f"Ganho: {GANHO:.2f}")

finally:
    camera.release()
    esp.close()
    cv2.destroyAllWindows()
