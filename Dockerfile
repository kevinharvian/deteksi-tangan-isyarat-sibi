# Gunakan Python 3.10 yang cocok dengan mediapipe
FROM python:3.10

# Set workdir
WORKDIR /app

# Copy semua file ke image
COPY . .

# Install dependensi
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Expose port yang dipakai Streamlit
EXPOSE 8501

# Jalankan Streamlit saat container start
CMD ["streamlit", "run", "streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
