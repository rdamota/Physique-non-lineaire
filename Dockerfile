FROM ubuntu:xenial
MAINTAINER Jean-Christophe Loiseau <loiseau.jc@gmail.com>
ENV DEBIAN_FRONTEND noninteractive

RUN apt-get update -q
RUN apt-get install -qy texlive-full 

