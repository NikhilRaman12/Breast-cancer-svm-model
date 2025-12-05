# 1. Use official Python base image
FROM python:3.10

# 2. Set working directory
WORKDIR /app

# 3. Copy entire project
COPY . .

# 4. Install dependencies
RUN pip install --default-timeout=100 --retries=10 --no-cache-dir -r requirements.txt

# 5. Run the Python app
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
