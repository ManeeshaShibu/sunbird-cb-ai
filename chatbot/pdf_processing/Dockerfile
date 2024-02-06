FROM python:3.9.17
WORKDIR /app
COPY . /app/
WORKDIR /app
RUN pip install --upgrade pip
RUN pip install -r requirements.txt
RUN pip install cython==0.29.36 numpy
RUN pip install sparse_dot_topn==0.3.0 --no-cache-dir --no-build-isolation
RUN pip install sparse-dot-topn-for-blocks --no-cache-dir --no-build-isolation
EXPOSE 5000
CMD ["python","qna_service.py"]