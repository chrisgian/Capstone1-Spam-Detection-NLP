FROM jupyter/minimal-notebook

WORKDIR /usr/src/app

ADD requirements.txt ./

RUN pip install --no-cache-dir -r requirements.txt
RUN easy_install gensim

ADD dl-stopword.py .
RUN python dl-stopword.py

ADD . . 

CMD python app.py