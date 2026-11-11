# Made for rdmt
FROM public.ecr.aws/lambda/python:3.13

# Copy rdmt-spire code to Lambda task directory in Docker container
COPY . ${LAMBDA_TASK_ROOT}

# pip install local rdmt-spire code with the correct version
ENV SETUPTOOLS_SCM_PRETEND_VERSION=0
RUN . ${LAMBDA_TASK_ROOT}/version.env && \
    SETUPTOOLS_SCM_PRETEND_VERSION=${PACKAGE_VERSION} && \
    pip3 install . --no-dependencies --no-cache-dir --target "${LAMBDA_TASK_ROOT}"

# pip install other reqired packages for the alembic update
RUN pip3 install pymysql sqlalchemy alembic numpy astropy --target "${LAMBDA_TASK_ROOT}"

COPY alembic_files/alembic_handler.py ${LAMBDA_TASK_ROOT}

# Set the alembic_handler to be the function that is executed when Docker container is started
CMD ["alembic_handler.handler"]