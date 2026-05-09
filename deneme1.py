from picamera2 import Picamera2
import cv2
import numpy as np
import time

# --- 1. İYİLEŞTİRME: FONKSİYON DÖNGÜ DIŞINA ALINDI ---
def kare_dikdortgen_mi(contour):
    """Açı kontrolü ile Üçgen, 5gen ve yamukları kesin eler."""
    area = cv2.contourArea(contour)
    if area < 50:
        return False, None

    epsilon = 0.05 * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)
    if len(approx) != 4:
        return False, None

    approx = approx.reshape(4, 2)
    cosines = []
    for i in range(4):
        p0 = approx[i]
        p1 = approx[(i + 1) % 4]
        p2 = approx[(i + 2) % 4]
        v1 = p0 - p1
        v2 = p2 - p1
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        if norm1 * norm2 == 0:
            return False, None
        cosine = abs(np.dot(v1, v2) / (norm1 * norm2))
        cosines.append(cosine)
        
    if max(cosines) > 0.55: 
        return False, None

    hull_area = cv2.contourArea(cv2.convexHull(contour))
    solidity = area / hull_area if hull_area > 0 else 0
    if solidity < 0.75:
        return False, None

    rect = cv2.minAreaRect(contour)
    (cx, cy), (w, h), _ = rect
    if w <= 0 or h <= 0:
        return False, None
    aspect_ratio = float(max(w, h)) / min(w, h)
    if aspect_ratio > 4.2:
        return False, None

    return True, rect


# --- MATEMATİKSEL OLASILIK SKORU ---
# 45m yükseklik, 640x480 çözünürlük, Pi Cam3 FOV=66°/41°
# Beklenen piksel kenar: mavi~86px, kırmızı~43px (merkez mesafe)
_BLUE_EXPECTED_PX = 86.0   # 4m hedef için piksel kenar tahmini
_RED_EXPECTED_PX  = 43.0   # 2m hedef için piksel kenar tahmini

def olasilik_skoru(contour, rect, renk_doygunlugu, beklenen_px):
    """
    0.0 – 1.0 arası matematiksel güven skoru döner.
    Dört alt kriter, ağırlıklı ortalama ile birleştirilir:
      - Şekil düzgünlüğü  (ağırlık 0.35): köşe açılarının 90°'e yakınlığı
      - Doluluk oranı     (ağırlık 0.25): alan / bounding-box alanı
      - Boyut uyumu       (ağırlık 0.25): gerçek dünya boyutuna yakınlık
      - Renk doygunluğu   (ağırlık 0.15): maskedeki ortalama S değeri
    """
    # --- 1. Şekil düzgünlüğü: köşe açıları 90°'e ne kadar yakın ---
    approx = cv2.approxPolyDP(contour, 0.05 * cv2.arcLength(contour, True), True)
    if len(approx) == 4:
        pts = approx.reshape(4, 2).astype(np.float32)
        max_cos = 0.0
        for i in range(4):
            v1 = pts[i] - pts[(i+1)%4]
            v2 = pts[(i+2)%4] - pts[(i+1)%4]
            n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
            if n1 * n2 > 0:
                max_cos = max(max_cos, abs(np.dot(v1, v2) / (n1 * n2)))
        # cos=0 → 90° → mükemmel kare, skor=1; cos=0.55 → skor≈0
        sekil_skoru = float(np.clip(1.0 - max_cos / 0.55, 0.0, 1.0))
    else:
        sekil_skoru = 0.3

    # --- 2. Doluluk oranı: contour alanı / bounding box alanı ---
    area = cv2.contourArea(contour)
    x, y, bw, bh = cv2.boundingRect(contour)
    bbox_area = bw * bh if bw * bh > 0 else 1
    doluluk = float(np.clip(area / bbox_area, 0.0, 1.0))

    # --- 3. Boyut uyumu: tespit edilen kenar ile beklenen pikseli karşılaştır ---
    _, (w, h), _ = rect
    tespit_px = (w + h) / 2.0
    oran = tespit_px / beklenen_px if beklenen_px > 0 else 1.0
    # 0.4x–2.5x arası kabul edilebilir, 1.0 mükemmel
    if oran < 1.0:
        boyut_skoru = float(np.clip(oran / 1.0, 0.0, 1.0))
    else:
        boyut_skoru = float(np.clip(1.0 - (oran - 1.0) / 1.5, 0.0, 1.0))

    # --- 4. Renk doygunluğu: 0-255 arasını 0-1'e normalize et ---
    renk_skoru = float(np.clip(renk_doygunlugu / 200.0, 0.0, 1.0))

    # --- Ağırlıklı toplam ---
    skor = (0.35 * sekil_skoru +
            0.25 * doluluk     +
            0.25 * boyut_skoru +
            0.15 * renk_skoru)
    return round(skor, 2)


# Picamera2 başlat - 480p
picam2 = Picamera2()
config = picam2.create_preview_configuration(main={"size": (640, 480)})
picam2.configure(config)
picam2.start()

time.sleep(2)  # kamera otursun

# --- 3. İYİLEŞTİRME: KULLANILMAYAN KERNEL SİLİNDİ ---

# --- 4. İYİLEŞTİRME: HIT COUNTER VE KİLİT DEĞİŞKENLERİ ---
REQUIRED_HITS = 10       # Hedefe kilitlenmek için gereken peş peşe tespit edilen kare sayısı (~0.3-0.4 sn)
TOLERANCE_FRAMES = 10    # Kilitlendikten sonra hedef anlık kaybolursa kilidi hemen bırakmamak için esneklik

red_hit_count = 0
red_lost_count = 0
red_locked = False

blue_hit_count = 0
blue_lost_count = 0
blue_locked = False

# FPS sayacı
prev_time = time.time()

# --- 2. İYİLEŞTİRME: HATA YÖNETİMİ (TRY-FINALLY) EKLENDİ ---
try:
    while True:
        frame = picam2.capture_array()

        # FPS hesapla
        curr_time = time.time()
        fps = 1.0 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 30
        prev_time = curr_time

        # Picamera2 RGB verir → OpenCV BGR ister
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        # Hafif blur
        blurred = cv2.blur(frame, (3, 3))

        # LAB renk uzayı
        lab = cv2.cvtColor(blurred, cv2.COLOR_BGR2LAB)

        # 🔴 KIRMIZI (LAB Uzayı)
        lower_red = np.array([20, 145, 110])
        upper_red = np.array([255, 255, 255])
        red_mask = cv2.inRange(lab, lower_red, upper_red)

        # 🔵 MAVİ (LAB Uzayı)
        lower_blue = np.array([20, 0, 0])
        upper_blue = np.array([255, 140, 115])
        blue_mask = cv2.inRange(lab, lower_blue, upper_blue)

        # Konturlar
        red_contours,  _ = cv2.findContours(red_mask,  cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        blue_contours, _ = cv2.findContours(blue_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        red_detected_this_frame = False
        blue_detected_this_frame = False

        # 🔴 KIRMIZI
        red_conf = 0.0
        for contour in red_contours:
            ok, rect = kare_dikdortgen_mi(contour)
            if ok:
                # Renk doygunluğu: kontur maskesindeki ortalama LAB a* kanalı
                mask_tmp = np.zeros(lab.shape[:2], dtype=np.uint8)
                cv2.drawContours(mask_tmp, [contour], -1, 255, -1)
                mean_a = float(cv2.mean(lab[:,:,1], mask=mask_tmp)[0])
                red_conf = olasilik_skoru(contour, rect, mean_a, _RED_EXPECTED_PX)
                box = np.int32(cv2.boxPoints(rect))
                cv2.drawContours(frame, [box], 0, (0, 0, 255), 2)
                cx, cy = int(rect[0][0]), int(rect[0][1])
                cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)
                cv2.putText(frame, f"%{int(red_conf*100)}", (cx+6, cy-6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)
                red_detected_this_frame = True
                break # En büyük/ilk bulunanı algılamak yeterli, diğerlerini aramaya gerek yok (Performans artışı)

        # 🔵 MAVİ
        blue_conf = 0.0
        for contour in blue_contours:
            ok, rect = kare_dikdortgen_mi(contour)
            if ok:
                # Renk doygunluğu: kontur maskesindeki ortalama LAB b* kanalı (mavi → düşük b*)
                mask_tmp = np.zeros(lab.shape[:2], dtype=np.uint8)
                cv2.drawContours(mask_tmp, [contour], -1, 255, -1)
                mean_b = float(cv2.mean(lab[:,:,2], mask=mask_tmp)[0])
                # b* 128 → nötr, <128 → mavi; renk doygunluğu = 128 - mean_b (ne kadar mavi o kadar yüksek)
                blue_doy = float(np.clip(128.0 - mean_b, 0, 128)) * (200.0 / 128.0)
                blue_conf = olasilik_skoru(contour, rect, blue_doy, _BLUE_EXPECTED_PX)
                box = np.int32(cv2.boxPoints(rect))
                cv2.drawContours(frame, [box], 0, (255, 0, 0), 2)
                cx, cy = int(rect[0][0]), int(rect[0][1])
                cv2.circle(frame, (cx, cy), 4, (255, 0, 0), -1)
                cv2.putText(frame, f"%{int(blue_conf*100)}", (cx+6, cy-6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 80, 0), 1)
                blue_detected_this_frame = True
                break

        # ==========================================
        #  KIRMIZI HIT COUNTER & LOCK-ON MANTIĞI
        # ==========================================
        if red_detected_this_frame:
            red_hit_count += 1
            red_lost_count = 0  # Gördüğümüz an kayıp sayacını sıfırla
            if red_hit_count >= REQUIRED_HITS:
                red_locked = True
        else:
            red_hit_count = 0 # Kesinti olursa hedef arama sayacı başa döner
            if red_locked:
                red_lost_count += 1
                # Eğer kilitliysek ve kayıp sayacı toleransı aşarsa kilidi bırak
                if red_lost_count > TOLERANCE_FRAMES:
                    red_locked = False

        # ==========================================
        #  MAVİ HIT COUNTER & LOCK-ON MANTIĞI
        # ==========================================
        if blue_detected_this_frame:
            blue_hit_count += 1
            blue_lost_count = 0
            if blue_hit_count >= REQUIRED_HITS:
                blue_locked = True
        else:
            blue_hit_count = 0
            if blue_locked:
                blue_lost_count += 1
                if blue_lost_count > TOLERANCE_FRAMES:
                    blue_locked = False

        # ==========================================
        #  DURUM EKRANA YAZDIRMA
        # ==========================================
        if red_locked:
            cv2.putText(frame, "[KILITLI] KIRMIZI HEDEF", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        elif red_hit_count > 0:
            # Yüzdelik olarak yüklenme barı gibi göster (%10, %20... %100)
            percent = int((red_hit_count / REQUIRED_HITS) * 100)
            cv2.putText(frame, f"KIRMIZI ARANIYOR... %{percent}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2) # Turuncu renk

        if blue_locked:
            cv2.putText(frame, "[KILITLI] MAVI HEDEF", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        elif blue_hit_count > 0:
            percent = int((blue_hit_count / REQUIRED_HITS) * 100)
            cv2.putText(frame, f"MAVI ARANIYOR... %{percent}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 165, 0), 2)

        # FPS - sağ üst köşe
        cv2.putText(frame, f"FPS: {fps:.1f}", (540, 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        cv2.imshow("Frame", frame)
        cv2.imshow("Kirmizi Maske", red_mask)
        cv2.imshow("Mavi Maske", blue_mask)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

except KeyboardInterrupt:
    print("\nKullanıcı tarafından durduruldu.")
except Exception as e:
    print(f"\nBir hata oluştu: {e}")
finally:
    # --- KAMERANIN GÜVENLE KAPANMASI ---
    cv2.destroyAllWindows()
    picam2.stop()
    print("Kamera ve pencereler başarıyla kapatıldı.")
