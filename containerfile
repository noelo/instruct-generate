FROM registry.access.redhat.com/ubi9/python-311

USER 1001

COPY requirements.txt .
RUN pip install --upgrade pip
# Example for installing on the Linux cpu-only version
RUN pip install docling --extra-index-url https://download.pytorch.org/whl/cpu
RUN pip install -r requirements.txt

RUN docling-tools models download

