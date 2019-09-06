FROM elasticsearch:7.3.1                                                                                                                                                
RUN yum install -y https://centos7.iuscommunity.org/ius-release.rpm
RUN yum update && \
    yum install -y \
        python36u \
        python36u-libs \
        python36u-devel \
        python36u-pip \
        git && \
    yum clean all
WORKDIR /home/elasticsearch/
RUN git clone https://github.com/jtibshirani/text-embeddings.git && \
    python3.6 -m pip install -r /home/elasticsearch/text-embeddings/requirements.txt
ENTRYPOINT ["python3.6"]
CMD ["/home/elasticsearch/text-embeddings/src/main.py"]
