from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')

#frases

frase1 = "El perro corre feliz por el parque"
frase2 = "El canino disfruta corriendo en la plaza"
frase3 = "Me gusta programar en python"

#Generar embeddings(vectores numericos)

emb1 = model.encode(frase1)
emb2 = model.encode(frase2)
emb3 = model.encode(frase3)

print(f"Longitud del vector: {len(emb1)}")  #dimensions..
print(f"Primeros 10 numeros del 1er vector: {emb1[:10]}")  #0.23, -0.45
print(f"Primeros 10 numeros del 2do vector: {emb2[:10]}")  #0.23, -0.45
print(f"Primeros 10 numeros del 3er vector: {emb3[:10]}")  #0.23, -0.45

