# Despliegue en EC2 (AWS)

Guía paso a paso para desplegar Poe-RAG en una instancia EC2, basada en un despliegue real (Ubuntu 26.04, sin GPU, 2 vCPU / 8GB RAM).

## Prerrequisitos

- Instancia EC2 con Ubuntu (probado en 26.04) y acceso SSH.
- Security Group con los puertos:
  - `22` (SSH)
  - `7860` (interfaz Gradio)
- API key de OpenAI.

## 1. Conectarse a la instancia

```bash
ssh -i "ruta/a/la-llave.pem" ubuntu@<host-publico-ec2>
```

## 2. Clonar el repositorio

```bash
git clone https://github.com/camilousa/por_rag.git
cd por_rag
```

## 3. Instalar Docker

```bash
sudo apt-get update -y
sudo apt-get install -y docker.io
sudo systemctl enable --now docker
sudo usermod -aG docker ubuntu
```

**Nota:** en algunas VPCs de AWS Academy Learner Lab, el mirror regional de Ubuntu (`us-east-1.ec2.archive.ubuntu.com`) no responde y `apt-get update` se queda colgado sin avanzar. Si esto ocurre, cambie al mirror genérico antes de reintentar:

```bash
sudo sed -i 's|us-east-1.ec2.archive.ubuntu.com|archive.ubuntu.com|g' /etc/apt/sources.list.d/ubuntu.sources
sudo apt-get update -y
```

## 4. Levantar Chroma y Ollama con Docker

```bash
sudo docker run -d --name poe_rag_chroma --restart unless-stopped \
  -p 8000:8000 -e IS_PERSISTENT=TRUE chromadb/chroma:1.3.5

sudo docker run -d --name poe_rag_ollama --restart unless-stopped \
  -p 11434:11434 -v poe_rag_ollama:/root/.ollama ollama/ollama
```

Descargar los modelos usados por el proyecto:

```bash
sudo docker exec poe_rag_ollama ollama pull embeddinggemma:latest
sudo docker exec poe_rag_ollama ollama pull llama3.2:3b
```

## 5. Preparar el entorno Python

**Nota importante:** Ubuntu 26.04 trae Python 3.14 por defecto. Varias dependencias del proyecto (numpy y el resto del stack de `requirements.txt`) todavía no publican wheels precompilados para esa versión, y compilarlas desde cero requiere `build-essential` y resulta lento y frágil. La solución que funcionó fue instalar Python 3.13 explícitamente mediante el PPA de `deadsnakes`:

```bash
sudo add-apt-repository -y ppa:deadsnakes/ppa
sudo apt-get update -y
sudo apt-get install -y python3.13 python3.13-venv
```

Crear el entorno virtual con esa versión:

```bash
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

Instalar dependencias. Para solo **servir** la app (chat + retriever, sin el stack pesado de ingesta/enriquecimiento/ML que no hace falta en producción), utilice el archivo `requirements-serve.txt` incluido en el repositorio:

```bash
pip install -r requirements-serve.txt
```

Si además se necesita correr la ingesta completa (`src/ingest/*`, evaluaciones con `ragas`, etc.), utilice el `requirements.txt` completo — en ese caso instale también `build-essential` y los headers de Python por si algún paquete necesita compilar:

```bash
sudo apt-get install -y build-essential python3.13-dev
pip install -r requirements.txt
```

## 6. Configurar `.env`

```bash
cat > .env <<'EOF'
OPENAI_API_KEY=la_api_key_de_openai
OPENAI_MODEL=gpt-4o-mini

CHROMA_HOST=127.0.0.1
CHROMA_PORT=8000
CHROMA_COLLECTION_NAME=poe_rag

OLLAMA_EMBED_BASE_URL=http://127.0.0.1:11434
OLLAMA_EMBED_MODEL=embeddinggemma:latest

OLLAMA_RERANK_BASE_URL=http://127.0.0.1:11434
OLLAMA_RERANK_MODEL=llama3.2:3b

RAGAS_LLM_DELAY=0.0
RATE_LIMIT_SECONDS=0.0
EOF
chmod 600 .env
```

**Nota sobre `MLFLOW_TRACKING_URI`:** solo debe incluirse en el `.env` si apunta a un servidor de MLflow realmente accesible desde esta instancia. Los decoradores `@mlflow.trace` del código leen esta variable automáticamente (aunque no se llame explícitamente a `mlflow.set_tracking_uri`), de modo que si apunta a un host inalcanzable, **cada llamada al chat queda colgada** reintentando la conexión. Si no hay un tracking server disponible, simplemente no debe definirse esta variable — MLflow usará tracking local en `./mlruns` sin hacer llamadas de red.

## 7. Indexar los datos en Chroma

```bash
python -m src.backend.vectorstore
```

Esto lee `data/gold/*.jsonl` (chunks ya enriquecidos), calcula embeddings con Ollama y los sube a la colección de Chroma.

## 8. Lanzar la aplicación

Para una prueba rápida:

```bash
nohup python -m src.frontend.gradio_app > gradio_app.log 2>&1 &
disown
```

La app queda escuchando en `0.0.0.0:7860`. Para que sobreviva a reinicios o desconexiones SSH de forma más robusta, se recomienda crear un servicio `systemd` en vez de `nohup`.

## 9. Abrir el puerto 7860 en el Security Group

Por consola de AWS (EC2 > Security Groups > el grupo correspondiente > Inbound rules > Add rule: TCP 7860, origen `0.0.0.0/0`), o por CLI:

```bash
aws ec2 authorize-security-group-ingress \
  --group-id <security-group-id> \
  --protocol tcp --port 7860 --cidr 0.0.0.0/0
```

## 10. Verificar

```bash
curl -I http://<ip-publica-ec2>:7860/
```

Y abrir `http://<ip-publica-ec2>:7860` en el navegador para probar el chat.

## Notas sobre AWS Academy Learner Lab

Si se utiliza una instancia de un Learner Lab (llave `vockey.pem`):

- Las credenciales de AWS son temporales y expiran cada pocas horas; deben refrescarse desde el panel "AWS Details" del lab.
- Al detener/iniciar la instancia, la IP pública normalmente cambia.
- Los recursos se eliminan cuando termina la sesión del lab — este despliegue no es persistente a largo plazo.
