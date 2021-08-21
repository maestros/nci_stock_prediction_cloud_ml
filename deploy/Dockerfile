FROM python:3.8.0-slim
COPY . /.
COPY requirements.txt /requirements.txt
RUN pip install -r requirements.txt
EXPOSE 9099
CMD ["python","predict.py"]