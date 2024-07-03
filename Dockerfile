FROM python:3.12
RUN useradd -m jupyter
EXPOSE 8888

RUN apt update && apt install -y lsof hdf5

# Install Python requirements
RUN pip install --upgrade --no-cache-dir hatch pip arviz

WORKDIR /jupyter
COPY --chown=1000:1000 ./ /jupyter
RUN chown -R 1000:1000 /jupyter/
RUN pip install -e /jupyter

# Switch to non-root user. It is crucial for security reasons to not run jupyter as root user!
USER jupyter

# Service
CMD ["python", "-m", "beaker_kernel.server.main", "--ip", "0.0.0.0"]
