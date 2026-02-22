"""CIF parsing and validation.

Parse Crystallographic Information Files (CIF) and validate their syntax.
The two main public functions are ``load_cif``, which parses a CIF and
returns extracted data items organised by data block, and ``validate_cif``,
which checks CIF syntax and reports the location of any errors.

Internally, parsing is performed by CIFParser (a sequential state machine)
and validation by CIFValidator (a line-by-line scanner). Both operate on
the raw text of the CIF file. CIFParser delegates to CIFValidator before
extracting data so that only syntactically valid files are parsed.

Classes:
    CIFParseError: Exception raised for all CIF syntax errors.
    DataBlock: Container for a single CIF data block.
    CIFParser: Parses a CIF file into DataBlock objects.
    CIFValidator: Validates CIF syntax line by line.
"""

import collections
import re
import warnings
from pathlib import Path
from re import Pattern

__all__ = ["CIFParseError", "load_cif", "validate_cif"]

DataItem = str | list[str]


def load_cif(filepath: str) -> dict[str, dict[str, DataItem]]:
    """Parse a CIF file and return all data items organised by data block.

    Read and parse the CIF, splitting the content into data blocks. For
    each data block, extract all data items (inline, semicolon, and loop)
    and return them in a nested dictionary keyed first by data block header
    and then by data name.

    Non-loop data items are stored as plain strings. Loop data items are
    stored as lists of strings, one per row.

    Args:
        filepath: Path to the CIF file to parse.

    Returns:
        A dictionary mapping each data block header to a dictionary of
        data name to data value (str or list[str]) pairs.

    Raises:
        CIFParseError: If the CIF contains a syntax error.

    Examples:
        Load a single-block CIF file:

        >>> from diffraction import load_cif
        >>> calcite_data = load_cif("calcite.cif")
        >>> calcite_data.keys()
        dict_keys(['data_calcite'])
        >>> calcite_data['data_calcite']['chemical_name_mineral']
        'Calcite'
        >>> calcite_data['data_calcite']['cell_volume']
        '366.63'

        Loop data items are returned as lists:

        >>> calcite_data['data_calcite']['atom_type_symbol']
        ['Ca2+', 'C4+', 'O2-']

        Multiple data blocks are each accessible by their header:

        >>> cif_data = load_cif("calcite_with_impurities.cif")
        >>> cif_data.keys()
        dict_keys(['data_calcite', 'data_aragonite', 'data_vaterite'])
    """
    if not filepath.lower().endswith(".cif"):
        warnings.warn(
            (
                "No .cif file extension detected. Assuming the filetype"
                "is CIF and continuing."
            ),
            UserWarning,
            stacklevel=2,
        )
    p = CIFParser(filepath)
    p.parse()
    return {data_block.header: data_block.data_items for data_block in p.data_blocks}


def validate_cif(filepath: str) -> bool:
    """Validate CIF syntax and report any errors with line numbers.

    Scan the CIF line by line and check for syntax errors. Return ``True``
    if no errors are found; raise ``CIFParseError`` with the line number
    and content if an error is detected.

    Args:
        filepath: Path to the CIF file to validate.

    Returns:
        True if the file syntax is valid.

    Raises:
        CIFParseError: If a syntax error is found, with a message indicating
            the error type, line number, and offending line content.

    Notes:
        Syntax errors detected include:
            - Empty file
            - Missing inline data name (lone data value)
            - Missing inline data value (lone data name)
            - Unmatched loop data names and data values
            - Unclosed semicolon text field

    Examples:
        >>> validate_cif("path/to/valid.cif")
        True
        >>> validate_cif("path/to/invalid.cif")
        CIFParseError: Missing inline data name on line 3: "some_lone_value"
    """
    if not filepath.lower().endswith(".cif"):
        warnings.warn(
            (
                "No .cif file extension detected. Assuming the filetype"
                "is CIF and continuing."
            ),
            UserWarning,
            stacklevel=2,
        )
    with Path(filepath).open() as cif_file:
        raw_data = cif_file.read()
    v = CIFValidator(raw_data)
    return v.validate()


# Regular expressions used for parsing.
COMMENT_OR_BLANK = re.compile(r"\w*#.*|\s+$|^$")
DATA_BLOCK_HEADER = re.compile("(?:^|\n)(data_\\S*)\\s*", re.IGNORECASE)
LOOP = re.compile("(?:^|\n)loop_\\s*", re.IGNORECASE)
DATA_NAME = re.compile(r"\s*_(\S+)")
DATA_NAME_START_LINE = re.compile("(?:^|\n)\\s*_(\\S+)")
DATA_VALUE = re.compile("\\s*('[^']+'|\"[^\"]+\"|[^\\s_#][^\\s'\"]*)")

DATA_VALUE_QUOTES = re.compile("^[\"']?(.*?)[\"']?$", re.DOTALL)
TEXT_FIELD = re.compile("[^_][^;]+")
SEMICOLON_DATA_ITEM = re.compile(
    f"(?:^|\n){DATA_NAME.pattern}\n;\n((?:.(?<!\n;))*)\n;", re.DOTALL
)
INLINE_DATA_ITEM = re.compile(
    f"(?:^|\n){DATA_NAME.pattern}[^\\S\n]+{DATA_VALUE.pattern}"
)


def strip_quotes(data_value: str) -> str:
    """Strip surrounding quotes from a CIF data value string."""
    return DATA_VALUE_QUOTES.match(data_value).group(1)


class DataBlock:
    """Container for one CIF data block's header and extracted data items.

    Store the raw CIF text for one data block and progressively extract
    data items from it. As each extraction method runs, the matched content
    is removed from ``raw_data`` so that subsequent extraction methods
    operate only on the remaining text.

    Args:
        header: The data block header string (e.g. ``'data_calcite'``).
        raw_data: Raw CIF text of this data block, excluding the header
            line and with comments and blank lines still present.

    Attributes:
        header: Data block header string.
        raw_data: Remaining raw CIF text (shrinks as items are extracted).
        data_items: Dictionary mapping data name strings to extracted values
            (str for scalar items, list[str] for loop items).
    """

    def __init__(self, header: str, raw_data: str) -> None:
        self.header = header
        self.raw_data = raw_data
        self.data_items = {}

    def extract_data_items(self, data_item_pattern: Pattern) -> None:
        """Extract non-loop data items matching a regex pattern.

        Find all matches of ``data_item_pattern`` in ``raw_data``, store
        each (data_name, data_value) pair in ``data_items``, then remove
        the matched text from ``raw_data``.

        Args:
            data_item_pattern: Compiled regex that matches a data item and
                captures the data name in group 1 and the data value in
                group 2. Used for inline and semicolon data items.

        Notes:
            The extraction is destructive: matched text is removed from
            ``raw_data``. Call semicolon extraction before inline extraction,
            and both before loop extraction.
        """
        data_items = data_item_pattern.findall(self.raw_data)
        for data_name, data_value in data_items:
            self.data_items[data_name] = strip_quotes(data_value)
        self.raw_data = data_item_pattern.sub("", self.raw_data)

    def extract_loop_data_items(self) -> None:
        """Extract all loop data items from the remaining raw data.

        For each loop in ``raw_data``, collect the data names declared at
        the loop header and then pair each subsequent row of data values
        with those names. The results are stored in ``data_items`` as
        lists, one value per row.

        The extracted data layout in ``data_items`` is::

            {
                "data_name_A": ["value_A1", "value_A2", ...],
                "data_name_B": ["value_B1", "value_B2", ...],
            }

        Notes:
            This method must be called last, after all non-loop items have
            been extracted. It assumes the remaining ``raw_data`` contains
            only loop constructs.

            Rows with fewer values than declared data names are silently
            skipped (zip strict=False), as partial rows occur legitimately
            in some CIF files.
        """
        loops = LOOP.split(self.raw_data)[1:]
        for loop in loops:
            data_names = DATA_NAME_START_LINE.findall(loop)
            for data_name in data_names:
                self.data_items[data_name] = []
            data_value_lines = loop.split("\n")[len(data_names) :]
            for line in data_value_lines:
                data_values = DATA_VALUE.findall(line)
                # zip with strict=False: data_values may be shorter than
                # data_names when a line has fewer values (partial loop rows
                # are skipped intentionally by this parser design)
                for data_name, data_value in zip(data_names, data_values, strict=False):
                    self.data_items[data_name].append(strip_quotes(data_value))

    def __repr__(self) -> str:
        """Return a representation of DataBlock, abbreviating raw data."""
        if len(self.raw_data) > 18:
            raw_data = f"{self.raw_data:.15s}..."
        else:
            raw_data = self.raw_data
        return f"DataBlock({self.header!r}, {raw_data!r}, {self.data_items!r})"

    def __eq__(self, other: "DataBlock") -> bool:
        return self.header == other.header and self.data_items == other.data_items


class CIFParser:
    """Parse a CIF file and extract data items for each data block.

    Read a CIF file, validate its syntax using CIFValidator, then split
    the content into DataBlock objects and extract all data items (inline,
    semicolon, and loop). The three extraction passes must run in order
    because each pass removes matched text from the raw data, and the loop
    extraction assumes all non-loop items have already been removed.

    Args:
        filepath: Path to the CIF file to parse.

    Attributes:
        raw_data: Raw file contents after reading (before stripping).
        data_blocks: List of DataBlock objects populated by ``parse()``.

    Examples:
        >>> p = CIFParser("path/to/cif.cif")
        >>> p.parse()
    """

    def __init__(self, filepath: str) -> None:
        with Path(filepath).open() as cif:
            self.raw_data = cif.read()
        validator = CIFValidator(self.raw_data)
        validator.validate()
        self.data_blocks = []

    def _strip_comments_and_blank_lines(self) -> None:
        """Remove all comment and blank lines from the raw file string."""
        lines = self.raw_data.split("\n")
        keep = [line for line in lines if not COMMENT_OR_BLANK.match(line)]
        self.raw_data = "\n".join(keep)

    def _extract_data_blocks(self) -> None:
        """Split the raw file string into DataBlock objects."""
        self.data_blocks = []
        data_blocks = DATA_BLOCK_HEADER.split(self.raw_data)[1:]
        headers, blocks = data_blocks[::2], data_blocks[1::2]
        # headers and blocks always come in pairs from the regex split
        for header, data in zip(headers, blocks, strict=True):
            self.data_blocks.append(DataBlock(header, data))

    def parse(self) -> None:
        """Parse the CIF and extract all data items from each data block.

        Strip comments and blank lines, split into data blocks, then for
        each data block extract semicolon items, inline items, and loop
        items in that order.

        Notes:
            The extraction order is mandatory. Each pass removes matched
            text from ``DataBlock.raw_data``. The loop extraction pass
            assumes only loop constructs remain in the raw data, so it
            must run last.
        """
        self._strip_comments_and_blank_lines()
        self._extract_data_blocks()
        for data_block in self.data_blocks:
            data_block.extract_data_items(SEMICOLON_DATA_ITEM)
            data_block.extract_data_items(INLINE_DATA_ITEM)
            data_block.extract_loop_data_items()


class CIFParseError(Exception):
    """Exception raised for CIF syntax errors."""


class CIFValidator:
    """Validate CIF syntax by scanning line by line.

    Perform a context-sensitive scan through the raw CIF text, checking
    for syntax errors at the top level, inside loops, and inside semicolon
    text fields. Raise CIFParseError with the offending line number and
    content if an error is found.

    Args:
        raw_data: The complete raw text content of the CIF file.

    Attributes:
        lines: Generator yielding successive lines of the CIF.
        current_line: The line currently being examined.
        line_number: Line number of ``current_line`` (1-based).

    Notes:
        Syntax errors detected include:
            - Empty file (warning only)
            - Missing inline data name (lone data value at top level)
            - Missing inline data value (lone data name not followed by
              a semicolon text field)
            - Unmatched loop data names and data values
            - Unclosed semicolon text field
    """

    def __init__(self, raw_data: str) -> None:
        """Initialise the validator on the first line of the CIF text.

        Warn if the file is empty, then split the raw data into a line
        generator and advance to the first line.

        Args:
            raw_data: The complete raw text content of the CIF file.
        """
        if not raw_data or raw_data.isspace():
            warnings.warn("File is empty.", stacklevel=2)
        self.lines = (line for line in raw_data.split("\n"))
        self.current_line = next(self.lines)
        self.line_number = 1

    def error(
        self,
        message: str | None = None,
        line_number: int | None = None,
        line: str | None = None,
    ) -> None:
        """Raise a CIFParseError with line number and content.

        Args:
            message: Error description string.
            line_number: Line number of the error; defaults to the current
                line number.
            line: Line content; defaults to the current line.

        Raises:
            CIFParseError: Always raised with the formatted error message.
        """
        if line_number is None:
            line_number, line = self.line_number, self.current_line
        raise CIFParseError(f'{message} on line {line_number}: "{line}"')

    def validate(self) -> bool:
        """Validate the CIF line by line.

        Perform a context-sensitive scan through the CIF, handling the
        top-level context, loop blocks, and semicolon text fields.

        Returns:
            True if the file syntax is valid.

        Raises:
            CIFParseError: If a syntax error is found in the CIF.
        """
        try:
            while True:
                if self._is_valid_single_line():
                    self._next_line()
                elif LOOP.match(self.current_line):
                    self._validate_loop()
                elif DATA_VALUE.match(self.current_line.lstrip()):
                    self.error("Missing inline data name")
                elif DATA_NAME.match(self.current_line):
                    self._validate_lone_data_name()
        except StopIteration:
            return True

    def _next_line(self) -> None:
        """Advance to the next line and increment the line counter."""
        self.current_line = next(self.lines)
        self.line_number += 1

    def _validate_loop(self) -> None:
        """Validate loop syntax.

        Collect loop data names, then verify that each subsequent data
        row has exactly the same number of values as there are data names.

        Raises:
            CIFParseError: If a data row has a different number of values
                than the number of declared loop data names.
        """
        loop_data_names = self._get_loop_data_names()
        while True:
            if COMMENT_OR_BLANK.match(self.current_line):
                self._next_line()
            elif self._is_loop_data_values():
                data_values = DATA_VALUE.findall(self.current_line)
                if len(data_values) != len(loop_data_names):
                    self.error("Unmatched data values to data names in loop")
                self._next_line()
            else:
                break

    def _get_loop_data_names(self) -> list[str]:
        """Collect data names declared at the start of a loop.

        Advance past the loop_ keyword, then gather all leading data
        name lines. Stop at the first line that is not a data name,
        comment, or blank line; that line is assumed to be the start of
        the loop data values.

        Returns:
            List of data name strings found in the loop header.
        """
        loop_data_names = []
        self._next_line()
        while True:
            if COMMENT_OR_BLANK.match(self.current_line):
                self._next_line()
            elif DATA_NAME.match(self.current_line):
                loop_data_names.append(DATA_NAME.match(self.current_line).group())
                self._next_line()
            else:
                break
        return loop_data_names

    def _validate_lone_data_name(self) -> None:
        """Validate an isolated data name.

        A lone data name either begins a semicolon data item (the next
        line starts with ``;``) or is a bare name with no value, which
        is a syntax error.

        Raises:
            CIFParseError: If the data name is not followed by a semicolon
                text field value.
        """
        err_line_number, err_line = self.line_number, self.current_line
        try:
            self._next_line()
        # check if final line of file
        except StopIteration:
            self.error("Invalid inline data value", err_line_number, err_line)
        # check if part of semicolon data item
        if self.current_line.startswith(";"):
            self._validate_semicolon_data_item()
        else:
            self.error("Invalid inline data value", err_line_number, err_line)

    def _validate_semicolon_data_item(self) -> None:
        """Validate a semicolon-delimited text field.

        Scan forward from the opening semicolon until a closing semicolon
        is found on its own line. Raise CIFParseError if end of file is
        reached without a closing semicolon.

        Raises:
            CIFParseError: If the semicolon text field has no closing ``;``.
        """
        self._next_line()
        # two line queue must be kept as if no closing semicolon is found,
        # then error occurred on previous line.
        previous_lines = collections.deque(maxlen=2)
        while True:
            if COMMENT_OR_BLANK.match(self.current_line) or TEXT_FIELD.match(
                self.current_line
            ):
                previous_lines.append((self.line_number, self.current_line))
                try:
                    self._next_line()
                # check if final line of file
                except StopIteration:
                    self.error("Unclosed semicolon text field")
            else:
                break
        if not self.current_line.startswith(";"):
            self.error("Unclosed semicolon text field", *previous_lines[1])
        self._next_line()

    def _is_valid_single_line(self) -> bool:
        """Check if the current line is valid and requires no further context.

        Return True for comment/blank lines, inline data items, and data
        block headers at the top level (i.e. not inside a loop or semicolon
        text field).

        Returns:
            True if the current line is self-contained and valid.
        """
        is_comment_or_blank = bool(COMMENT_OR_BLANK.match(self.current_line))
        is_inline_data_item = bool(INLINE_DATA_ITEM.match(self.current_line))
        is_data_block_header = bool(DATA_BLOCK_HEADER.match(self.current_line))

        return is_comment_or_blank or is_inline_data_item or is_data_block_header

    def _is_loop_data_values(self) -> bool:
        """Check if the current line contains data values in a loop context.

        Returns:
            True if the line starts with a data value and is neither a
            loop_ keyword nor a data block header.
        """
        return (
            DATA_VALUE.match(self.current_line)
            and not LOOP.match(self.current_line)
            and not DATA_BLOCK_HEADER.match(self.current_line)
        )
