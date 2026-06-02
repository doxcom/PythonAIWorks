import qrcode

# Datos que quieres convertir en QR
datos = input("¿Qué texto o enlace quieres convertir a QR? ")

# Crear el código QR
qr = qrcode.QRCode(version=1, box_size=10, border=5)
qr.add_data(datos)
qr.make(fit=True)

# Generar y guardar la imagen
img = qr.make_image(fill_color="black", back_color="white")
img.save("mi_primer_qr.png")

print("¡QR generado! Busca 'mi_primer_qr.png' en la carpeta actual")