import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Carregar e redimensionar a imagem
img = cv2.imread("/home/lesc/Documentos/Bancos de dados/FIVE/train/Original/65_A.png")
img = cv2.resize(img, (512, 521))

# Separar os 3 canais de cores (BGR - padrão OpenCV)
b, g, r = cv2.split(img)

# ---------- FUNÇÃO PARA CALCULAR SOBEL COMPLETO EM UM CANAL ----------
def sobel_completo_canal(canal, nome_canal):
    """
    Calcula o Sobel em todas as direções para um canal específico
    """
    # Converter para float32 para melhor precisão
    canal_float = canal.astype(np.float32)
    
    # ---------- SOBEL PADRÃO (Horizontal e Vertical) ----------
    sobel_x = cv2.Sobel(canal_float, cv2.CV_32F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(canal_float, cv2.CV_32F, 0, 1, ksize=3)
    
    # ---------- SOBEL DIAGONAL (45° e 135°) ----------
    kernel_45 = np.array([[-2, -1, 0],
                          [-1,  0, 1],
                          [ 0,  1, 2]], dtype=np.float32)
    
    kernel_135 = np.array([[ 0, -1, -2],
                           [ 1,  0, -1],
                           [ 2,  1,  0]], dtype=np.float32)
    
    sobel_45 = cv2.filter2D(canal_float, cv2.CV_32F, kernel_45)
    sobel_135 = cv2.filter2D(canal_float, cv2.CV_32F, kernel_135)
    
    # ---------- COMBINAR TODAS AS DIREÇÕES ----------
    # Magnitude completa (todas as direções)
    sobel_completo = np.sqrt(sobel_x**2 + sobel_y**2 + sobel_45**2 + sobel_135**2)
    sobel_completo = cv2.convertScaleAbs(sobel_completo)
    
    # Componentes individuais para visualização
    sobel_x_abs = cv2.convertScaleAbs(sobel_x)
    sobel_y_abs = cv2.convertScaleAbs(sobel_y)
    sobel_45_abs = cv2.convertScaleAbs(sobel_45)
    sobel_135_abs = cv2.convertScaleAbs(sobel_135)
    
    # Combinar apenas H+V (tradicional)
    sobel_hv = np.sqrt(sobel_x**2 + sobel_y**2)
    sobel_hv = cv2.convertScaleAbs(sobel_hv)
    
    # Combinar apenas diagonais
    sobel_diag = np.sqrt(sobel_45**2 + sobel_135**2)
    sobel_diag = cv2.convertScaleAbs(sobel_diag)
    
    return {
        'nome': nome_canal,
        'completo': sobel_completo,
        'hv': sobel_hv,
        'diag': sobel_diag,
        'x': sobel_x_abs,
        'y': sobel_y_abs,
        '45': sobel_45_abs,
        '135': sobel_135_abs,
        'original': canal
    }

# ---------- PROCESSAR OS 3 CANAIS ----------
print("🔄 Processando canais de cores...")
resultados_r = sobel_completo_canal(r, 'Vermelho (R)')
resultados_g = sobel_completo_canal(g, 'Verde (G)')
resultados_b = sobel_completo_canal(b, 'Azul (B)')
print("✅ Processamento concluído!")

# ---------- COMBINAR OS 3 CANAIS ----------
# Método 1: Média dos 3 canais
sobel_completo_rgb_media = cv2.addWeighted(
    cv2.addWeighted(resultados_r['completo'], 0.33, 
                   resultados_g['completo'], 0.33, 0),
    1.0, resultados_b['completo'], 0.34, 0
)

# Método 2: Máximo dos 3 canais (preserva bordas mais fortes)
sobel_completo_rgb_max = np.maximum(
    np.maximum(resultados_r['completo'], resultados_g['completo']),
    resultados_b['completo']
)

# Método 3: Soma normalizada (maior intensidade)
sobel_completo_rgb_soma = cv2.add(
    cv2.add(resultados_r['completo'], resultados_g['completo']),
    resultados_b['completo']
)
sobel_completo_rgb_soma = cv2.normalize(sobel_completo_rgb_soma, None, 0, 255, cv2.NORM_MINMAX)
'''
# ---------- VISUALIZAÇÃO 1: Canais Individuais ----------
fig, axs = plt.subplots(3, 4, figsize=(20, 15))

# Canal Vermelho (R)
axs[0, 0].imshow(resultados_r['original'], cmap='Reds')
axs[0, 0].set_title(f'Canal {resultados_r["nome"]} - Original', fontsize=12)
axs[0, 0].axis('off')

axs[0, 1].imshow(resultados_r['x'], cmap='gray')
axs[0, 1].set_title('Sobel X (Horizontal)', fontsize=12)
axs[0, 1].axis('off')

axs[0, 2].imshow(resultados_r['y'], cmap='gray')
axs[0, 2].set_title('Sobel Y (Vertical)', fontsize=12)
axs[0, 2].axis('off')

axs[0, 3].imshow(resultados_r['completo'], cmap='Reds')
axs[0, 3].set_title('Sobel Completo (Todas Direções)', fontsize=12)
axs[0, 3].axis('off')

# Canal Verde (G)
axs[1, 0].imshow(resultados_g['original'], cmap='Greens')
axs[1, 0].set_title(f'Canal {resultados_g["nome"]} - Original', fontsize=12)
axs[1, 0].axis('off')

axs[1, 1].imshow(resultados_g['x'], cmap='gray')
axs[1, 1].set_title('Sobel X (Horizontal)', fontsize=12)
axs[1, 1].axis('off')

axs[1, 2].imshow(resultados_g['y'], cmap='gray')
axs[1, 2].set_title('Sobel Y (Vertical)', fontsize=12)
axs[1, 2].axis('off')

axs[1, 3].imshow(resultados_g['completo'], cmap='Greens')
axs[1, 3].set_title('Sobel Completo (Todas Direções)', fontsize=12)
axs[1, 3].axis('off')

# Canal Azul (B)
axs[2, 0].imshow(resultados_b['original'], cmap='Blues')
axs[2, 0].set_title(f'Canal {resultados_b["nome"]} - Original', fontsize=12)
axs[2, 0].axis('off')

axs[2, 1].imshow(resultados_b['x'], cmap='gray')
axs[2, 1].set_title('Sobel X (Horizontal)', fontsize=12)
axs[2, 1].axis('off')

axs[2, 2].imshow(resultados_b['y'], cmap='gray')
axs[2, 2].set_title('Sobel Y (Vertical)', fontsize=12)
axs[2, 2].axis('off')

axs[2, 3].imshow(resultados_b['completo'], cmap='Blues')
axs[2, 3].set_title('Sobel Completo (Todas Direções)', fontsize=12)
axs[2, 3].axis('off')

plt.tight_layout()
plt.show()

# ---------- VISUALIZAÇÃO 2: Combinação dos 3 Canais ----------
fig, axs = plt.subplots(2, 3, figsize=(18, 10))

# Imagem original colorida
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
axs[0, 0].imshow(img_rgb)
axs[0, 0].set_title('Imagem Original (RGB)', fontsize=14)
axs[0, 0].axis('off')

# Canal Vermelho completo
axs[0, 1].imshow(resultados_r['completo'], cmap='Reds')
axs[0, 1].set_title('Sobel Completo - Canal Vermelho', fontsize=14)
axs[0, 1].axis('off')

# Canal Verde completo
axs[0, 2].imshow(resultados_g['completo'], cmap='Greens')
axs[0, 2].set_title('Sobel Completo - Canal Verde', fontsize=14)
axs[0, 2].axis('off')

# Canal Azul completo
axs[1, 0].imshow(resultados_b['completo'], cmap='Blues')
axs[1, 0].set_title('Sobel Completo - Canal Azul', fontsize=14)
axs[1, 0].axis('off')

# Combinação: Média dos 3 canais
axs[1, 1].imshow(sobel_completo_rgb_media, cmap='gray')
axs[1, 1].set_title('Combinação (Média dos 3 Canais)', fontsize=14)
axs[1, 1].axis('off')

# Combinação: Máximo dos 3 canais
axs[1, 2].imshow(sobel_completo_rgb_max, cmap='gray')
axs[1, 2].set_title('Combinação (Máximo dos 3 Canais)', fontsize=14)
axs[1, 2].axis('off')

plt.tight_layout()
plt.show()

# ---------- VISUALIZAÇÃO 3: Comparação Detalhada ----------
fig, axs = plt.subplots(2, 4, figsize=(20, 10))

# Linha 1: Sobel H+V para cada canal
axs[0, 0].imshow(resultados_r['hv'], cmap='Reds')
axs[0, 0].set_title('R - H+V (Tradicional)', fontsize=12)
axs[0, 0].axis('off')

axs[0, 1].imshow(resultados_g['hv'], cmap='Greens')
axs[0, 1].set_title('G - H+V (Tradicional)', fontsize=12)
axs[0, 1].axis('off')

axs[0, 2].imshow(resultados_b['hv'], cmap='Blues')
axs[0, 2].set_title('B - H+V (Tradicional)', fontsize=12)
axs[0, 2].axis('off')

axs[0, 3].imshow(cv2.addWeighted(
    cv2.addWeighted(resultados_r['hv'], 0.33, resultados_g['hv'], 0.33, 0),
    1.0, resultados_b['hv'], 0.34, 0), cmap='gray')
axs[0, 3].set_title('RGB - H+V (Média)', fontsize=12)
axs[0, 3].axis('off')

# Linha 2: Sobel completo (todas direções) para cada canal
axs[1, 0].imshow(resultados_r['completo'], cmap='Reds')
axs[1, 0].set_title('R - Completo (Todas Direções)', fontsize=12)
axs[1, 0].axis('off')

axs[1, 1].imshow(resultados_g['completo'], cmap='Greens')
axs[1, 1].set_title('G - Completo (Todas Direções)', fontsize=12)
axs[1, 1].axis('off')

axs[1, 2].imshow(resultados_b['completo'], cmap='Blues')
axs[1, 2].set_title('B - Completo (Todas Direções)', fontsize=12)
axs[1, 2].axis('off')

axs[1, 3].imshow(sobel_completo_rgb_media, cmap='gray')
axs[1, 3].set_title('RGB - Completo (Média)', fontsize=12)
axs[1, 3].axis('off')

plt.tight_layout()
plt.show()
'''
# ---------- ANÁLISE QUANTITATIVA ----------
print("\n" + "="*60)
print("📊 ANÁLISE QUANTITATIVA DOS 3 CANAIS")
print("="*60)

canais = [
    ('Vermelho (R)', resultados_r),
    ('Verde (G)', resultados_g),
    ('Azul (B)', resultados_b)
]

for nome, res in canais:
    print(f"\n🔴 Canal {nome}:")
    print(f"  Média H+V: {np.mean(res['hv']):.2f}")
    print(f"  Média Diagonais: {np.mean(res['diag']):.2f}")
    print(f"  Média Completo: {np.mean(res['completo']):.2f}")
    print(f"  Desvio padrão: {np.std(res['completo']):.2f}")
    print(f"  Bordas fortes (>100): {np.sum(res['completo'] > 100)} pixels")

print(f"\n📈 Combinação dos 3 canais (Média):")
print(f"  Média: {np.mean(sobel_completo_rgb_media):.2f}")
print(f"  Desvio padrão: {np.std(sobel_completo_rgb_media):.2f}")
print(f"  Bordas fortes (>100): {np.sum(sobel_completo_rgb_media > 100)} pixels")
'''
# ---------- HISTOGRAMAS DOS 3 CANAIS ----------
fig, axs = plt.subplots(1, 3, figsize=(18, 5))

# Histograma do Canal Vermelho
axs[0].hist(resultados_r['completo'].ravel(), bins=256, color='red', alpha=0.7)
axs[0].set_title('Histograma - Canal Vermelho', fontsize=14)
axs[0].set_xlabel('Intensidade da Borda')
axs[0].set_ylabel('Frequência')

# Histograma do Canal Verde
axs[1].hist(resultados_g['completo'].ravel(), bins=256, color='green', alpha=0.7)
axs[1].set_title('Histograma - Canal Verde', fontsize=14)
axs[1].set_xlabel('Intensidade da Borda')
axs[1].set_ylabel('Frequência')

# Histograma do Canal Azul
axs[2].hist(resultados_b['completo'].ravel(), bins=256, color='blue', alpha=0.7)
axs[2].set_title('Histograma - Canal Azul', fontsize=14)
axs[2].set_xlabel('Intensidade da Borda')
axs[2].set_ylabel('Frequência')

plt.tight_layout()
plt.show()

# ---------- VISUALIZAÇÃO 3D DOS 3 CANAIS ----------
fig = plt.figure(figsize=(18, 6))

# Subamostragem para visualização 3D
step = 8
x = np.arange(0, sobel_completo_rgb_media.shape[1], step)
y = np.arange(0, sobel_completo_rgb_media.shape[0], step)
X, Y = np.meshgrid(x, y)

# Canal Vermelho
ax1 = fig.add_subplot(131, projection='3d')
Z_r = resultados_r['completo'][::step, ::step]
ax1.plot_surface(X, Y, Z_r, cmap='Reds', alpha=0.8)
ax1.set_title('Canal Vermelho - 3D', fontsize=12)
ax1.set_xlabel('X')
ax1.set_ylabel('Y')
ax1.set_zlabel('Intensidade')

# Canal Verde
ax2 = fig.add_subplot(132, projection='3d')
Z_g = resultados_g['completo'][::step, ::step]
ax2.plot_surface(X, Y, Z_g, cmap='Greens', alpha=0.8)
ax2.set_title('Canal Verde - 3D', fontsize=12)
ax2.set_xlabel('X')
ax2.set_ylabel('Y')
ax2.set_zlabel('Intensidade')

# Canal Azul
ax3 = fig.add_subplot(133, projection='3d')
Z_b = resultados_b['completo'][::step, ::step]
ax3.plot_surface(X, Y, Z_b, cmap='Blues', alpha=0.8)
ax3.set_title('Canal Azul - 3D', fontsize=12)
ax3.set_xlabel('X')
ax3.set_ylabel('Y')
ax3.set_zlabel('Intensidade')

plt.tight_layout()
plt.show()
'''
# ---------- SALVAR RESULTADOS ----------
print("\n💾 Salvando resultados...")

# Salvar cada canal individualmente
cv2.imwrite("sobel_R_completo.png", resultados_r['completo'])
cv2.imwrite("sobel_G_completo.png", resultados_g['completo'])
cv2.imwrite("sobel_B_completo.png", resultados_b['completo'])

# Salvar combinações
cv2.imwrite("sobel_RGB_media.png", sobel_completo_rgb_media)
cv2.imwrite("sobel_RGB_max.png", sobel_completo_rgb_max)
cv2.imwrite("sobel_RGB_soma.png", sobel_completo_rgb_soma)
sobel_soma_suavisado= cv2.bilateralFilter(sobel_completo_rgb_soma, 9, 75, 75)
cv2.imwrite("sobel_soma_suavisado.png", sobel_soma_suavisado)
sobel_soma_canny=cv2.Canny(sobel_soma_suavisado,threshold1=60,threshold2=100)
cv2.imwrite("sobel_soma_canny.png", sobel_soma_canny)
cv2.imwrite("sobel_RGB_soma_equalizado.png", cv2.equalizeHist(sobel_completo_rgb_soma))


# Salvar imagem original processada (como no seu código)
# Inverter a imagem combinada para manter compatibilidade
sobel_invertida = cv2.bitwise_not(sobel_completo_rgb_media)
cv2.imwrite("teste.png", sobel_invertida)

print("✅ Todas as imagens salvas com sucesso!")
print("📁 Arquivos gerados:")
print("   - sobel_R_completo.png (Canal Vermelho)")
print("   - sobel_G_completo.png (Canal Verde)")
print("   - sobel_B_completo.png (Canal Azul)")
print("   - sobel_RGB_media.png (Média dos 3 canais)")
print("   - sobel_RGB_max.png (Máximo dos 3 canais)")
print("   - sobel_RGB_soma.png (Soma normalizada)")
print("   - teste.png (Invertida - compatível com seu código)")

print("\n🎯 RESULTADO FINAL:")
print("="*60)
print("✅ Processamento completo dos 3 canais de cores")
print("✅ Detecção de bordas em TODAS as direções (H, V, 45°, 135°)")
print("✅ 3 métodos de combinação dos canais: Média, Máximo e Soma")
print("✅ Visualizações individuais e combinadas")
print("✅ Análise estatística de cada canal")