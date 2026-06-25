import cv2
import numpy as np
import matplotlib.pyplot as plt

# Carregar e redimensionar a imagem
img = cv2.imread("/home/lesc/Documentos/Bancos de dados/FIVE/train/Original/65_A.png")

# Converter para escala de cinza para processamento
img_cinza = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# ---------- SOBEL PADRÃO (Horizontal e Vertical) ----------
sobel_x = cv2.Sobel(img_cinza, cv2.CV_32F, 1, 0, ksize=3)
sobel_y = cv2.Sobel(img_cinza, cv2.CV_32F, 0, 1, ksize=3)

# ---------- SOBEL DIAGONAL (45° e 135°) ----------
# Kernels para diagonais
kernel_45 = np.array([[-2, -1, 0],
                      [-1,  0, 1],
                      [ 0,  1, 2]], dtype=np.float32)

kernel_135 = np.array([[ 0, -1, -2],
                       [ 1,  0, -1],
                       [ 2,  1,  0]], dtype=np.float32)

# Aplicar os kernels diagonais na imagem em escala de cinza
sobel_45 = cv2.filter2D(img_cinza, cv2.CV_32F, kernel_45)
sobel_135 = cv2.filter2D(img_cinza, cv2.CV_32F, kernel_135)

# ---------- COMBINAR TODAS AS DIREÇÕES ----------
# Método 1: Raiz quadrada da soma dos quadrados (mais preciso)
sobel_completo = np.sqrt(sobel_x**2 + sobel_y**2 + sobel_45**2 + sobel_135**2)
sobel_completo = cv2.convertScaleAbs(sobel_completo)

# Método 2: Soma ponderada (alternativa)
# sobel_completo = cv2.convertScaleAbs(sobel_x) + cv2.convertScaleAbs(sobel_y) + cv2.convertScaleAbs(sobel_45) + cv2.convertScaleAbs(sobel_135)

# ---------- PROCESSAMENTO INDIVIDUAL PARA VISUALIZAÇÃO ----------
# Converter para uint8 para exibição
sobel_x_abs = cv2.convertScaleAbs(sobel_x)
sobel_y_abs = cv2.convertScaleAbs(sobel_y)
sobel_45_abs = cv2.convertScaleAbs(sobel_45)
sobel_135_abs = cv2.convertScaleAbs(sobel_135)

# Combinar apenas horizontal e vertical (Sobel tradicional)
sobel_hv = np.sqrt(sobel_x**2 + sobel_y**2)
sobel_hv = cv2.convertScaleAbs(sobel_hv)

# Combinar apenas diagonais
sobel_diag = np.sqrt(sobel_45**2 + sobel_135**2)
sobel_diag = cv2.convertScaleAbs(sobel_diag)

# ---------- PROCESSAMENTO ADICIONAL (SEU CÓDIGO ORIGINAL) ----------
# Para manter a compatibilidade com seu código original
# Vou criar versões em BGR para as visualizações
img_bgr = img.copy()

# Converter resultados para BGR (3 canais) para manter o padrão do seu código
sobel_x_bgr = cv2.cvtColor(sobel_x_abs, cv2.COLOR_GRAY2BGR)
sobel_y_bgr = cv2.cvtColor(sobel_y_abs, cv2.COLOR_GRAY2BGR)
sobel_hv_bgr = cv2.cvtColor(sobel_hv, cv2.COLOR_GRAY2BGR)
sobel_completo_bgr = cv2.cvtColor(sobel_completo, cv2.COLOR_GRAY2BGR)
 
# Processamento em escala de cinza
sobel_cinza = cv2.cvtColor(sobel_invertida, cv2.COLOR_BGR2GRAY)
sobel_cinza = cv2.convertScaleAbs(sobel_cinza)

# Threshold adaptativo
sobel_binario = cv2.adaptiveThreshold(sobel_cinza, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                      cv2.THRESH_BINARY, 11, 2)

# Canny
sobel_canny = cv2.Canny(sobel_cinza, threshold1=10, threshold2=255)

# Equalização de histograma
sobel_equalizado = cv2.equalizeHist(sobel_cinza)

# Histogramas
hist = cv2.calcHist([sobel_cinza], [0], None, [256], [0, 256])
hist_sobel_equalizado = cv2.calcHist([sobel_equalizado], [0], None, [256], [0, 256])

# ---------- VISUALIZAÇÃO ----------
fig, axs = plt.subplots(2, 4, figsize=(15, 9))

# Primeira linha: componentes individuais
axs[0, 0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
axs[0, 0].set_title("Imagem Original")
axs[0, 0].axis("off")

axs[0, 1].imshow(sobel_x_abs, cmap='gray')
axs[0, 1].set_title("Sobel X (Horizontal)")
axs[0, 1].axis("off")

axs[0, 2].imshow(sobel_y_abs, cmap='gray')
axs[0, 2].set_title("Sobel Y (Vertical)")
axs[0, 2].axis("off")

axs[0, 3].imshow(sobel_hv, cmap='gray')
axs[0, 3].set_title("Sobel H+V (Tradicional)")
axs[0, 3].axis("off")

# Segunda linha: diagonais e combinação completa
axs[1, 0].imshow(sobel_45_abs, cmap='gray')
axs[1, 0].set_title("Sobel Diagonal 45°")
axs[1, 0].axis("off")

axs[1, 1].imshow(sobel_135_abs, cmap='gray')
axs[1, 1].set_title("Sobel Diagonal 135°")
axs[1, 1].axis("off")

axs[1, 2].imshow(sobel_diag, cmap='gray')
axs[1, 2].set_title("Sobel Diagonais (45°+135°)")
axs[1, 2].axis("off")

axs[1, 3].imshow(sobel_completo, cmap='gray')
axs[1, 3].set_title("Sobel COMPLETO (Todas Direções)")
axs[1, 3].axis("off")

plt.tight_layout()
plt.show()

# ---------- VISUALIZAÇÃO EXTRA: Comparação ----------
fig, axs = plt.subplots(1, 3, figsize=(15, 5))

axs[0].imshow(sobel_hv, cmap='gray')
axs[0].set_title("Apenas Horizontal + Vertical", fontsize=14)
axs[0].axis("off")

axs[1].imshow(sobel_diag, cmap='gray')
axs[1].set_title("Apenas Diagonais (45° + 135°)", fontsize=14)
axs[1].axis("off")

axs[2].imshow(sobel_completo, cmap='gray')
axs[2].set_title("TODAS Direções (H+V+Diagonais)", fontsize=14)
axs[2].axis("off")

plt.tight_layout()
plt.show()

# ---------- SALVAR RESULTADOS ----------
# Salvar a imagem invertida (como no código original)
sobel_invertida_bgr = sobel_invertida  # já está em BGR
cv2.imwrite("teste.png", sobel_invertida_bgr)

# Salvar todos os resultados
cv2.imwrite("sobel_horizontal.png", sobel_x_abs)
cv2.imwrite("sobel_vertical.png", sobel_y_abs)
cv2.imwrite("sobel_hv.png", sobel_hv)
cv2.imwrite("sobel_45.png", sobel_45_abs)
cv2.imwrite("sobel_135.png", sobel_135_abs)
cv2.imwrite("sobel_diagonal.png", sobel_diag)
cv2.imwrite("sobel_completo.png", sobel_completo)

print("✅ Todas as imagens salvas com sucesso!")
print(f"✅ Sobel completo combina: Horizontal + Vertical + 45° + 135°")

# Análise quantitativa da intensidade das bordas
print("="*50)
print("ANÁLISE DE INTENSIDADE DAS BORDAS")
print("="*50)

print(f"Média Sobel H+V: {np.mean(sobel_hv):.2f}")
print(f"Média Sobel Diagonais: {np.mean(sobel_diag):.2f}")
print(f"Média Sobel Completo: {np.mean(sobel_completo):.2f}")
print(f"Desvio padrão Sobel Completo: {np.std(sobel_completo):.2f}")

# Mostrar quantas bordas a mais foram detectadas
bordas_hv = np.sum(sobel_hv > 50)
bordas_diag = np.sum(sobel_diag > 50)
bordas_completo = np.sum(sobel_completo > 50)

print(f"\nBordas detectadas (threshold > 50):")
print(f"H+V: {bordas_hv} pixels")
print(f"Diagonais: {bordas_diag} pixels")
print(f"Completo: {bordas_completo} pixels")
print(f"Aumento: {(bordas_completo/bordas_hv - 1)*100:.1f}% mais bordas")

from mpl_toolkits.mplot3d import Axes3D

# Criar uma grade para visualização 3D
x = np.arange(0, sobel_completo.shape[1], 5)
y = np.arange(0, sobel_completo.shape[0], 5)
X, Y = np.meshgrid(x, y)
Z = sobel_completo[::5, ::5]

fig = plt.figure(figsize=(15, 10))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, Z, cmap='plasma', alpha=0.8)
ax.set_title('Magnitude do Gradiente - Sobel Completo (Todas Direções)', fontsize=14)
ax.set_xlabel('X (pixels)')
ax.set_ylabel('Y (pixels)')
ax.set_zlabel('Intensidade da Borda')
plt.show()