FROM nvidia/cuda:12.6.3-devel-ubuntu24.04

ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update \
    && apt-get install -y --no-install-recommends python3 ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Prepared outside the image from one exact mining commit. SOURCE_ARCHIVE is a
# git archive; PILOT_ARCHIVE contains S10 plus its required S6/S7 kernel input.
COPY source.tar.gz pilot-model.tar.gz SHA256SUMS PACKAGE.sha256 /opt/qtc-e/
COPY rental_entrypoint.sh /opt/qtc-e/rental_entrypoint.sh
RUN chmod 0555 /opt/qtc-e/rental_entrypoint.sh \
    && cd /opt/qtc-e \
    && sha256sum --check --strict PACKAGE.sha256 \
    && sha256sum --check --strict SHA256SUMS

EXPOSE 21559/tcp
ENTRYPOINT ["/opt/qtc-e/rental_entrypoint.sh"]
