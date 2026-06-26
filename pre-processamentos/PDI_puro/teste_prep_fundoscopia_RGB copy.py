import cv2
import numpy as np
import matplotlib.pyplot as plt 
#from bm3d import bm3d
import cv2
import numpy as np

def equalizar_colorida_hsv(imagem):
 
    # Converter BGR para HSV
    hsv = cv2.cvtColor(imagem, cv2.COLOR_BGR2HSV)
    
    # Separar canais
    h, s, v = cv2.split(hsv)
    
    # Equalizar apenas o canal V (Value/Intensidade)
    v_equalizado = cv2.equalizeHist(v)
    
    # Juntar novamente
    hsv_equalizado = cv2.merge([h, s, v_equalizado])
    
    # Converter de volta para BGR
    resultado = cv2.cvtColor(hsv_equalizado, cv2.COLOR_HSV2BGR)
    
    return resultado
def equalizar_com_correcao_cor(imagem):
    """
    Equaliza mantendo o equilíbrio de cores
    """
    # Equalizar HSV
    hsv = cv2.cvtColor(imagem, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    v_eq = cv2.equalizeHist(v)
    hsv_eq = cv2.merge([h, s, v_eq])
    img_eq = cv2.cvtColor(hsv_eq, cv2.COLOR_HSV2BGR)
    
    # Corrigir saturação exagerada
    hsv_eq2 = cv2.cvtColor(img_eq, cv2.COLOR_BGR2HSV)
    h2, s2, v2 = cv2.split(hsv_eq2)
    
    # Limitar saturação para não estourar
    s2 = np.clip(s2, 0, 200)
    
    hsv_final = cv2.merge([h2, s2, v2])
    resultado = cv2.cvtColor(hsv_final, cv2.COLOR_HSV2BGR)
    
    return resultado
 

img = cv2.imread("/home/emanuel/Documentos/mestrado/bases de dados/FIVES/train/Original/65_A.png")
img = cv2.resize(img,(1080,1080))

img_equalizada = equalizar_com_correcao_cor(img)
cv2.imshow("img_equalizada", img_equalizada)
cv2.waitKey(0)
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

hist_sobel_equalizado = cv2.calcHist([sobel_equalizado], [0], None, [256], [0, 256])

 
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
  
sobel_invertida_ciza= cv2.cvtColor(sobel_invertida,cv2.COLOR_BGR2GRAY)
cv2.imshow("sobel_invertida_ciza", sobel_invertida_ciza)
cv2.waitKey(0)

hist = cv2.calcHist([sobel_invertida_ciza], [0], None, [256], [0, 256])

# Plotando o histograma usando o módulo pyplot do matplotlib
plt.figure()
plt.title("Grayscale Histogram")  # Definindo o título do gráfico
plt.xlabel("Bins")  # Definindo o rótulo do eixo X
plt.ylabel("# of Pixels")  # Definindo o rótulo do eixo Y
plt.plot(hist)  # Plotando o histograma
plt.xlim([0, 256])  # Limitando o eixo X de 0 a 256
plt.show()  # Exibindo o gráfico do histograma
 

sobel_invertida_ciza_EQUALIZADA= cv2.equalizeHist(sobel_invertida_ciza)
#cv2.imshow("sobel_invertida_ciza_EQUALIZADA", sobel_invertida_ciza_EQUALIZADA)
#cv2.waitKey(0)

hist = cv2.calcHist([sobel_invertida_ciza_EQUALIZADA], [0], None, [256], [0, 256])

# Plotando o histograma usando o módulo pyplot do matplotlib
plt.figure()
plt.title("sobel_invertida_ciza_EQUALIZADA")  # Definindo o título do gráfico
plt.xlabel("Bins")  # Definindo o rótulo do eixo X
plt.ylabel("# of Pixels")  # Definindo o rótulo do eixo Y
plt.plot(hist)  # Plotando o histograma
plt.xlim([0, 256])  # Limitando o eixo X de 0 a 256
plt.show()  # Exibindo o gráfico do histograma

'''
img_float = sobel_invertida_ciza.astype(np.float32)/255.0

# Desvio padrão estimado do ruído
sigma = 25/255

denoise = bm3d(img_float, sigma_psd=sigma)

denoise = (denoise*255).astype(np.uint8)

denoise_EQUALIZADA= cv2.equalizeHist(denoise)
 
cv2.imshow("Original", sobel_invertida_ciza_EQUALIZADA)
cv2.imshow("BM3D", denoise)1
 
fig, axs = plt.subplots(1, 2, figsize=(16, 5))

axs[0].imshow(denoise)
axs[0].set_title("denoise")
axs[0].axis("off")

axs[1].imshow(denoise_EQUALIZADA)
axs[1].set_title("denoise_EQUALIZADA")
axs[1].axis("off")


plt.tight_layout()
plt.show()
 '''
imagem_rgb = img.copy()#cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# 4. Separar os canais
R = imagem_rgb[:, :, 0]  # Canal Vermelho
G = imagem_rgb[:, :, 1]  # Canal Verde
B = imagem_rgb[:, :, 2]  # Canal Azul

# 5. Criar figuras para cada canal
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Imagem original
axes[0, 0].imshow(imagem_rgb)
axes[0, 0].set_title('Imagem Original')
axes[0, 0].axis('off')

# Canal Vermelho
axes[0, 1].imshow(R, cmap='Reds')
axes[0, 1].set_title('Canal Vermelho')
axes[0, 1].axis('off')

# Canal Verde
axes[1, 0].imshow(G, cmap='Greens')
axes[1, 0].set_title('Canal Verde')
axes[1, 0].axis('off')

# Canal Azul
axes[1, 1].imshow(B, cmap='Blues')
axes[1, 1].set_title('Canal Azul')
axes[1, 1].axis('off')
'''
plt.tight_layout()
plt.show()
'''
 
# Criar imagens com apenas um canal ativo
R_imagem = imagem_rgb.copy()
R_imagem[:, :, 1] = 0  # Zera verde
R_imagem[:, :, 2] = 0  # Zera azul

G_imagem = imagem_rgb.copy()
G_imagem[:, :, 0] = 0  # Zera vermelho
G_imagem[:, :, 2] = 0  # Zera azul

B_imagem = imagem_rgb.copy()
B_imagem[:, :, 0] = 0  # Zera vermelho
B_imagem[:, :, 1] = 0  # Zera verde

# Plotar
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

axes[0, 0].imshow(imagem_rgb)
axes[0, 0].set_title('Imagem Original')
axes[0, 0].axis('off')

axes[0, 1].imshow(R_imagem)
axes[0, 1].set_title('Canal Vermelho')
axes[0, 1].axis('off')

axes[1, 0].imshow(G_imagem)
axes[1, 0].set_title('Canal Verde')
axes[1, 0].axis('off')

axes[1, 1].imshow(B_imagem)
axes[1, 1].set_title('Canal Azul')
axes[1, 1].axis('off')


fig, axes = plt.subplots(2, 2, figsize=(12, 10))
R_imagem=cv2.cvtColor(R_imagem,cv2.COLOR_RGB2GRAY)
G_imagem=cv2.cvtColor(G_imagem,cv2.COLOR_RGB2GRAY)
B_imagem=cv2.cvtColor(B_imagem,cv2.COLOR_RGB2GRAY)
axes[0, 0].imshow(imagem_rgb)
axes[0, 0].set_title('Imagem Original')
axes[0, 0].axis('off')

axes[0, 1].imshow(R_imagem)
axes[0, 1].set_title('Canal Vermelho')
axes[0, 1].axis('off')

axes[1, 0].imshow(G_imagem)
axes[1, 0].set_title('Canal Verde')
axes[1, 0].axis('off')

axes[1, 1].imshow(B_imagem)
axes[1, 1].set_title('Canal Azul')
axes[1, 1].axis('off')




R_imagem=cv2.equalizeHist(R_imagem)
G_imagem=cv2.equalizeHist(G_imagem)
B_imagem=cv2.equalizeHist(B_imagem)
axes[0, 0].imshow(imagem_rgb)
axes[0, 0].set_title('Imagem Original')
axes[0, 0].axis('off')

axes[0, 1].imshow(R_imagem)
axes[0, 1].set_title('Canal Vermelho')
axes[0, 1].axis('off')

axes[1, 0].imshow(G_imagem)
axes[1, 0].set_title('Canal Verde')
axes[1, 0].axis('off')

axes[1, 1].imshow(B_imagem)
axes[1, 1].set_title('Canal Azul')
axes[1, 1].axis('off')






R_imagem=cv2.equalizeHist(R_imagem)


sobel_x = cv2.Sobel(R_imagem, cv2.CV_32F, 1, 0, ksize=1)
sobel_y = cv2.Sobel(R_imagem, cv2.CV_32F, 0, 1, ksize=3)

sobel = np.sqrt(sobel_x**2 + sobel_y**2)
r_sobel = cv2.convertScaleAbs(sobel)
 


G_imagem=cv2.equalizeHist(G_imagem)
sobel_x = cv2.Sobel(G_imagem, cv2.CV_32F, 1, 0, ksize=1)
sobel_y = cv2.Sobel(G_imagem, cv2.CV_32F, 0, 1, ksize=3)

sobel = np.sqrt(sobel_x**2 + sobel_y**2)
G_sobel = cv2.convertScaleAbs(sobel)



B_imagem=cv2.equalizeHist(B_imagem)
sobel_x = cv2.Sobel(B_imagem, cv2.CV_32F, 1, 0, ksize=1)
sobel_y = cv2.Sobel(B_imagem, cv2.CV_32F, 0, 1, ksize=3)

sobel = np.sqrt(sobel_x**2 + sobel_y**2)
B_sobel = cv2.convertScaleAbs(sobel)
axes[0, 0].imshow(imagem_rgb)
axes[0, 0].set_title('Imagem Original')
axes[0, 0].axis('off')

axes[0, 1].imshow(r_sobel)
axes[0, 1].set_title('Canal Vermelho SOBEL')
axes[0, 1].axis('off')

axes[1, 0].imshow(G_sobel)
axes[1, 0].set_title('Canal Verde SOBEL')
axes[1, 0].axis('off')

axes[1, 1].imshow(B_sobel)
axes[1, 1].set_title('Canal Azul SOBEL')
axes[1, 1].axis('off')





plt.tight_layout()
plt.show()