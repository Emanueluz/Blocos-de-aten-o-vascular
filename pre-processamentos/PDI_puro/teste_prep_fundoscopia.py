import cv2
import numpy as np
import matplotlib.pyplot as plt 


img = cv2.imread("/home/lesc/Documentos/Bancos de dados/FIVE/train/Original/65_A.png")
img = cv2.resize(img,(512,521))


sobel_x = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3)
sobel_y = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3)

sobel = np.sqrt(sobel_x**2 + sobel_y**2)
sobel = cv2.convertScaleAbs(sobel)
img_cinza = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)




 
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
sobel_x = cv2.cvtColor(sobel_x, cv2.COLOR_BGR2RGB)
sobel_y = cv2.cvtColor(sobel_y, cv2.COLOR_BGR2RGB)
sobel = cv2.cvtColor(sobel, cv2.COLOR_BGR2RGB)
sobel_invertida = cv2.bitwise_not(sobel )
sobel_cinza =  cv2.convertScaleAbs(cv2.cvtColor(sobel_invertida, cv2.COLOR_RGB2GRAY))
sobel_binario=cv2.adaptiveThreshold( sobel_cinza,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                    cv2.THRESH_BINARY,11,2)

sobel_canny= cv2.Canny(sobel_cinza, threshold1=10, threshold2=255)
sobel_equalizado=cv2.equalizeHist(sobel_cinza)
cv2.imshow("sobel_equalizado", sobel_equalizado)
cv2.imshow("sobel cinza", sobel_cinza)
cv2.imshow("sobel_canny", sobel_canny)
cv2.waitKey(0)


hist = cv2.calcHist([sobel_cinza], [0], None, [256], [0, 256])

plt.figure(figsize=(8,4))
plt.plot(hist)
plt.title("Histograma")
plt.xlabel("Intensidade")
plt.ylabel("Quantidade de Pixels")
plt.show()


sobel_invertida = np.uint8(np.absolute(sobel_invertida))

sobel_invertida = cv2.bitwise_not(sobel_invertida)

sobel_invertida = cv2.bitwise_not(sobel)

 
sobel_invertida= cv2.cvtColor(sobel_invertida,cv2.COLOR_RGB2BGR)
cv2.imwrite("teste.png", sobel_invertida)
fig, axs = plt.subplots(1, 5, figsize=(16, 5))

axs[0].imshow(img)
axs[0].set_title("imagem original")
axs[0].axis("off")

axs[1].imshow(sobel_x)
axs[1].set_title("sobel_x")
axs[1].axis("off")

axs[2].imshow(sobel_y)
axs[2].set_title("sobel_y")
axs[2].axis("off")

axs[3].imshow(sobel)
axs[3].set_title("sobel")
axs[3].axis("off")

axs[4].imshow(sobel_invertida)
axs[4].set_title("sobel_invertida")
axs[4].axis("off")

plt.tight_layout()
plt.show()
  
