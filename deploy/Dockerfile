 FROM python:3.9.4
 COPY requirements.txt /requirements.txt
 RUN pip install -r requirements.txt
 EXPOSE 9099
 COPY . /.
 CMD ["python","predict.py"]