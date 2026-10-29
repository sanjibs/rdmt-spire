from enum import IntEnum


# Define all the status codes
class StatusCodes(IntEnum):
    SUCCESS = 200
    FAILURE = 500 # TODO: 500 typically means an error on the server side rather than just a failure in the code itself
    BAD_EVENT_FORMAT = 422
    NOT_YET_IMPLEMENTED = 700
    BAD_LAMBDA_BATCH_SIZE = 701

    ALEMBIC_FAILED_VERSIONS_DOWNLOAD = 601
    ALEMBIC_UPDATE_FAILURE = 602

    # ingest problems
    REPROCESS_NUM_DETERMINATION_FAIL = 603
    UNEXPECTED_FILE_TYPE = 604
    SQS_SEND_FAIL = 605
    UNEXPECTED_EXPOSURE_TYPE = 606
