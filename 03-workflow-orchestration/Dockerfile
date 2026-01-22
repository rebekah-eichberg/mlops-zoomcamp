FROM apache/airflow:2.9.3

# Copy requirements into the image at build time
COPY requirements.txt /tmp/requirements.txt

# Install python deps as the airflow user (required by the Airflow image)
USER airflow
RUN pip install --no-cache-dir -r /tmp/requirements.txt