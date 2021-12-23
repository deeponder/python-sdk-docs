class AppPlatformError(Exception):
    """
    Raised by :meth:`Client.request()` for requests that:
      - Return a non-200 HTTP response, or
      - Connection refused/timeout or
      - Response timeout or
      - Malformed request
      - Have a malformed/missing header in the response.
    """

    def __init__(self, exc_message, status_code, error_code=None, json=None):
        super(AppPlatformError, self).__init__(exc_message)
        self.status_code = status_code
        self.error_code = error_code
        self.json = json or {}


class ServerError(AppPlatformError):
    """
    For 500-level responses from the server
    """


class ClientError(AppPlatformError):
    """
    For 400-level responses from the server
    has json parameter for additional information to be stored about error
    if need be
    """


class InputNotUnderstoodError(Exception):
    """
    Raised if a method is called in a way that cannot be understood
    """


class InvalidUsageError(Exception):
    """Raised when methods are called with invalid or incompatible arguments"""


class AllRetriesFailedError(Exception):
    """Raised when the retry manager does not successfully make a request"""
    

class PendingJobFinished(Exception):
    """
    Raised when the server responds with a 303 for the pending creation of a
    resource.
    """


class JobNotFinished(Exception):
    """
    Raised when execution was trying to get a finished resource from a pending
    job, but the job is not finished
    """


class UploadTrainDataError(Exception):
    """

    """

class GetProjectInfoError(Exception):
    """

    """

class ProjectTrainError(Exception):
    """

    """

class UploadDatasetError(Exception):
    """

    """