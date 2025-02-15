# syntax=docker/dockerfile:1.5-labs
ARG BASE_IMAGE=ubuntu:20.04
FROM ${BASE_IMAGE}
LABEL maintainer="harry0789@qq.com"

# build iEDA
ARG IEDA_REPO=.
ARG IEDA_WORKSPACE=/opt/iEDA
ARG iEDA_BINARY_DIR=${IEDA_WORKSPACE}/bin

ENV IEDA_WORKSPACE=${IEDA_WORKSPACE}
ENV PATH=${iEDA_BINARY_DIR}:${PATH}
ENV TZ=Asia/Shanghai

# (docker build) --ssh default=$HOME/.ssh/id_rsa
ADD ${IEDA_REPO} ${IEDA_WORKSPACE}

RUN ln -sf /usr/share/zoneinfo/${TZ} /etc/localtime && \
    bash ${IEDA_WORKSPACE}/build.sh -i mirror && \
    apt-get autoremove -y && apt-get clean -y && \
    bash ${IEDA_WORKSPACE}/build.sh -b ${iEDA_BINARY_DIR} && \
    rm -rf ${IEDA_WORKSPACE}/build \
    ${IEDA_WORKSPACE}/src/database/manager/parser/vcd/vcd_parser/target \
    ${IEDA_WORKSPACE}/src/database/manager/parser/verilog/verilog-rust/verilog-parser/target \
    ${IEDA_WORKSPACE}/src/database/manager/parser/spef/spef-parser/target \
    ${IEDA_WORKSPACE}/src/database/manager/parser/liberty/lib-rust/liberty-parser/target

WORKDIR ${IEDA_WORKSPACE}

CMD ["/usr/bin/env", "bash", "-c", "iEDA -script scripts/hello.tcl"]
