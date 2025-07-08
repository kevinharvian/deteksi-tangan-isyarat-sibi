# Menggunakan image dasar Python 3.10 versi slim (lebih ringan dari versi penuh)
FROM python:3.10-slim

# Menentukan direktori kerja di dalam container
WORKDIR /app

# Menyalin file requirements.txt terlebih dahulu untuk memanfaatkan caching Docker
COPY requirements.txt .

# Install dependensi sistem yang diperlukan oleh mediapipe & opencv
RUN apt-get update && apt-get install -y \
    build-essential \          
    cmake \                    
    libglib2.0-0 \             
    libsm6 \                    
    libxext6 \                  
    libxrender-dev \           
    && pip install --upgrade pip \                           
    && pip install --no-cache-dir -r requirements.txt \       
    && apt-get clean \                                        
    && rm -rf /var/lib/apt/lists/*                           

# Salin seluruh isi project ke direktori kerja container (/app)
COPY . .

# Buka port 8501 untuk Streamlit
EXPOSE 8501

# Perintah untuk menjalankan aplikasi Streamlit saat container dijalankan
CMD ["streamlit", "run", "streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
