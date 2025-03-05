FROM python:3.12

LABEL maintainer="Docater"  

WORKDIR /usr/src/app  

COPY requirements.txt ./  
RUN pip install -r requirements.txt  

COPY . .  

EXPOSE 8535  

CMD ["streamlit", "run", "app.py"]  