# Gunakan Python 3.10 yang cocok dengan mediapipe
FROM python:3.10-bullseye

# Set workdir
WORKDIR /app

# Copy requirements dulu biar bisa cache layer pip install
COPY requirements.txt .

# Install dependensi
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Copy semua file ke image
COPY . .

# Expose port yang dipakai Streamlit
EXPOSE 8501

# Jalankan Streamlit saat container start
CMD ["streamlit", "run", "streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
