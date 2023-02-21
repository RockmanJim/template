FROM python:3.7

WORKDIR /code
COPY ./ .

RUN pip3 config set global.index-url  http://mirrors.myhuaweicloud.com/pypi/web/simple \
    && pip3 config set install.trusted-host mirrors.myhuaweicloud.com
RUN pip install -r requirements.txt

CMD [ "bash" ]
