import cv2
import numpy as np
import matplotlib.pyplot as plt


if __name__ == '__main__':
    img = cv2.imread("./slika.png")
    # OpenCV ucitava sliku u formatu BGR
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # pretvaranje slike u nijanse sive (grayscale)

    center = (256, 256)
    # slika je 512x512 pa ce centar biti na poziciji (256, 256)
    radius = 50
    # radius idealnog low ili high pass filtra

    img_fft = np.fft.fft2(img)
    # pretvaranje slike u frekventni domen (FFT - Fast Fourier Transform), fft2 je jer je u 2 dimenzije
    img_fft = np.fft.fftshift(img_fft)

    img_fft_mag = np.abs(img_fft)
    # slika u frekventnom domenu je kompleksan broj, nama je potrebna amplituda tog kompleksnog broja
    # (odnosno moduo sto daje funkcija np.abs())

    img_mag_1 = img_fft / img_fft_mag
    # cuvanje kompleksnih brojeva sa jedinicnim moduom, jer cemo da menjamo amplitudu

    img_fft_log = np.log(img_fft_mag)
    # vrednosti su prevelike da bi se menjale direktno, pa je logaritam dobar nacin da se vizualizuje
    # amplituda frekventnog domena.

    #plt.imshow(img_fft_log, cmap='gray')
    #plt.show()

    plt.figure(figsize=(15, 5))

    plt.subplot(141)
    plt.imshow(img, cmap='gray')

    plt.subplot(142)
    plt.imshow(img_fft_log)

    img_fft_log[231, 251] = 0
    img_fft_log[246, 261] = 0
    img_fft_log[266, 251] = 0
    img_fft_log[281, 261] = 0
    img_fft = img_mag_1 * np.exp(img_fft_log)
    # vracamo magnitudu iz logaritma i mnozimo sa kompleksnim brojevima na slici
    img_filtered = np.abs(np.fft.ifft2(img_fft))
    # funkcija ifft2 vraca sliku iz frekventnog u prostorni domen, nije potrebno raditi ifftshift jer to se
    # implicitno izvrsava
    # rezultat ifft2 je opet kompleksna slika, ali nas zanima samo moduo jer to je nasa slika zato opet treba np.abs()

    plt.subplot(143)
    plt.imshow(img_fft_log)

    plt.subplot(144)
    plt.imshow(img_filtered, cmap='gray')
    plt.show()