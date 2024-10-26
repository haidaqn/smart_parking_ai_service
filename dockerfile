FROM python:3.12

RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    && apt-get clean

WORKDIR /app

COPY requirements.txt .
RUN python -m venv venv && \
    . venv/bin/activate && \
    pip install --upgrade pip && \
    pip install -r requirements.txt

COPY . .

CMD ["bash", "-c", ". venv/bin/activate && python app.py"]
