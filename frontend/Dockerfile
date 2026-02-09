FROM python:3.11-slim

WORKDIR /app

# 1. Copy shared requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 2. Copy frontend code
COPY frontend/ ./frontend/

# 3. Expose Chainlit Port
EXPOSE 8001

# 4. Start Chainlit
CMD ["chainlit", "run", "frontend/chainlit_app.py", "--host", "0.0.0.0", "--port", "8001"]