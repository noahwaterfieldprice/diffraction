"""Exception hierarchy for the diffraction package."""


class DiffractionError(Exception):
    """Base exception for all diffraction library errors.

    Callers can catch this to handle any error raised by the
    diffraction package without also catching unrelated ValueError
    or TypeError from other code.
    """


class SpaceGroupError(DiffractionError, ValueError):
    """Raised for invalid space group lookups.

    Inherits from both :class:`DiffractionError` and :class:`ValueError`
    so callers may catch either.
    """


class ScatteringDataError(DiffractionError, ValueError):
    """Raised for invalid element lookups or missing scattering data.

    Inherits from both :class:`DiffractionError` and :class:`ValueError`
    so callers may catch either.
    """
