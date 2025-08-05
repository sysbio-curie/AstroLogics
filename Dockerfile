FROM colomoto/colomoto-docker:2025-03-01
MAINTAINER Vincent Noel <contact@vincent-noel.fr>

USER root
RUN apt-get -qq update \
    && apt-get install -yq git \
    && apt clean -y \
    && rm -rf /var/lib/apt/lists/*

#RUN pip install git+https://https://github.com/saran-pankaew/AstroLogics.git
RUN mkdir -p /notebook/AstroLogics/
COPY Tutorial/* /notebook/AstroLogics/

RUN chown -R user:user /notebook/AstroLogics

USER user
