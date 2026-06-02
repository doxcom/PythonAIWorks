import qrcode
import random
import string

# Datos que quieres convertir en QR
datos = input("¿Qué texto o enlace quieres convertir a QR? ")

# Crear el código QR
qr = qrcode.QRCode(version=1, box_size=10, border=5)
qr.add_data(datos)
qr.make(fit=True)

# Generar y guardar la imagen
img = qr.make_image(fill_color="black", back_color="white")
randomName = ''.join(random.choices(string.ascii_letters + string.digits, k =10))
fileNameQR = randomName + ".png"
img.save(fileNameQR)

print(f"¡QR generado! Busca {fileNameQR} en la carpeta actual")