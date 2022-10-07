FROM public.ecr.aws/lambda/python:3.8

#RUN apt-get update && apt-get upgrade -y

RUN yum install mesa-libGL -y

COPY app/requirements.txt ${LAMBDA_TASK_ROOT}
RUN pip install -r requirements.txt

COPY app ${LAMBDA_TASK_ROOT}

CMD [ "app.handler" ]