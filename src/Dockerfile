FROM conda/miniconda3:latest

RUN conda install -c anaconda --yes tensorflow==1.14.0 \
    && conda install -c anaconda --yes opencv==3.4.2 \
    && conda install -c anaconda --yes pandas==0.25.0 \
    && conda install -c anaconda --yes click==7.0 \
    && conda install -c anaconda --yes scikit-learn==0.21.2 \
    && conda install -c conda-forge --yes pillow==6.1.0 \
    && conda install --yes ipython=7.7.0 \
    && conda install --yes memory_profiler=0.55.0 \
    && conda install --yes jupyter==1.0.0 \
    && conda install -c conda-forge --yes watermark==1.8.2 \
    && conda install -c conda-forge --yes matplotlib==3.1.1 \
    && pip install imutils==0.5.2 

COPY . /srv/