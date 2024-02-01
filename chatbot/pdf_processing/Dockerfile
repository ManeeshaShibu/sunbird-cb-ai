FROM python:3.9.17
WORKDIR /app
COPY . /app/
WORKDIR /app
RUN pip install --upgrade pip
RUN pip install -r requirements.txt
EXPOSE 5000
CMD ["python","qna_service.py"]