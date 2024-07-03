FROM python:3.13.0b2-slim

WORKDIR /app

COPY . .

# Install Chromium. Those are public API keys that even Debian uses.
RUN echo "ttf-mscorefonts-installer msttcorefonts/accepted-mscorefonts-eula select true" | debconf-set-selections && \
    sed -i -e "s/ main[[:space:]]*\$/ main contrib non-free/" /etc/apt/sources.list.d/debian.sources && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
      fontconfig \
      fonts-freefont-ttf \
      fonts-gfs-neohellenic \
      fonts-indic \
      fonts-ipafont-gothic \
      fonts-kacst \
      fonts-liberation \
      fonts-noto-cjk \
      fonts-noto-color-emoji \
      fonts-roboto \
      fonts-hanazono \
      fonts-thai-tlwg \
      fonts-wqy-zenhei \
      ttf-mscorefonts-installer \
      chromium

RUN fc-cache -f -v && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* /var/cache/apt/archives/* /tmp/* /var/tmp/* /usr/share/fonts/truetype/noto && \
    mkdir -p /etc/chromium.d/ && \
    echo -e 'export GOOGLE_API_KEY="AIzaSyCkfPOPZXDKNn8hhgu3JrA62wIgC93d44k"\nexport GOOGLE_DEFAULT_CLIENT_ID="811574891467.apps.googleusercontent.com"\nexport GOOGLE_DEFAULT_CLIENT_SECRET="kdloedMFGdGla2P1zacGjAQh"' > /etc/chromium.d/googleapikeys

# Add u2net.onnx model for the remove background library
ADD https://github.com/danielgatis/rembg/releases/download/v0.0.0/u2net.onnx /root/.u2net/u2net.onnx

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

CMD ["python", "chatbot.py"]