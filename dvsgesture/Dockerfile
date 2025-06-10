FROM python:3.12-slim
RUN pip install tensorboard
WORKDIR /logs
EXPOSE 6006
CMD ["tensorboard", "--logdir=.", "--host=0.0.0.0"]
