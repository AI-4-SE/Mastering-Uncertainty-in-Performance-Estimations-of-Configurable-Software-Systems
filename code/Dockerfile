# Note: Since git repositories are cloned, an active internet connection is required

# The predictions were performed on Debian 9 (stretch)
FROM python:3.7-slim-buster

# Set the working directory to /app
WORKDIR /application
RUN mkdir -p /results/last-inference/
RUN mkdir -p /results/last-evaluation/

COPY . .

# install tools to install dependencies
RUN apt update
RUN apt install -y -qq git wget unzip tar

#
RUN git clone --depth=1 https://github.com/ChristianKaltenecker/Distance-Based_Data.git \
    && tar -xzf Distance-Based_Data/SupplementaryWebsite/MeasuredPerformanceValues/JavaGC/measurements.tar.gz -C Distance-Based_Data/SupplementaryWebsite/MeasuredPerformanceValues/JavaGC/ \
    && tar -xzf Distance-Based_Data/SupplementaryWebsite/MeasuredPerformanceValues/VP9/measurements.tar.gz -C Distance-Based_Data/SupplementaryWebsite/MeasuredPerformanceValues/VP9/

RUN pip3 install https://github.com/pymc-devs/pymc3/archive/v3.8.tar.gz

RUN pip3 install ./activesampler
RUN pip3 install ./p4


# /application/Distance-Based_Data/SupplementaryWebsite/MeasuredPerformanceValues