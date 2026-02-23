"""Exception hierarchy for the diffraction package."""


class DiffractionError(Exception):
    """Base exception for all diffraction library errors.

    Callers can catch this to handle any error raised by the
    diffraction package without also catching unrelated ValueError
    or TypeError from other code.
    """
