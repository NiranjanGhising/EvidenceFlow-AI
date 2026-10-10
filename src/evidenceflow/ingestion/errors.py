"""Custom exceptions raised by the ingestion pipeline."""


class IngestionError(Exception):
    """Base exception for document ingestion failures."""


class UnsupportedFileTypeError(IngestionError):
    """Raised when a document type is not supported."""


class FileTooLargeError(IngestionError):
    """Raised when a document exceeds the configured size limit."""


class EmptyDocumentError(IngestionError):
    """Raised when no usable text can be extracted."""


class UnsafePathError(IngestionError):
    """Raised when a path does not point to a safe regular file."""