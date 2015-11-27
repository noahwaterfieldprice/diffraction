"""CIF file parsing and validation

Module contains three main functions `load_cif`, `validate_cif` and `cif2json`.

Functions
---------
load_cif:
    Extract :term:`data items` from :term:`CIF` file and return a
    dictionary of :term:`data blocks`.
validate_cif:
    Validate :term:`CIF` file, and return True if no syntax errors
    are found, else raise a :class:`CIFParseError`.
cif2json:
    Extract :term:`data items` from :term:`CIF` file and save to a
    JSON file with :term:`data items` sorted alphabetically by name.

Attributes
----------
CIFParseError:
    Exception for all parse errors due to incorrect :term:`CIF` syntax.


"""  # TODO: finish this docstring

import json
import re
import warnings
from collections import OrderedDict, deque

__all__ = ["load_cif", "validate_cif", "CIFParseError"]


def load_cif(filepath):
    """Extract and return :term:`data items` from :term:`CIF` file.

    Input CIF file is read and split by :term:`data block`. The
    :term:`data block header` and data items are extracted for each
    data block and returned in dictionary. Each one is represented by
    a `key: value` pair where the key is the data block header and
    the value is dictionary of the corresponding data items.

    Within each **data block**:

        Each non-:term:`loop` data item is stored in a key: value pair
        where the key and value are the :term:`data name` and
        :term:`data value` respectively.

        Loop data is stored in a child dictionary under the key
        `"loop_i"` where i is the number of the loop in the order they
        appear in the CIF. Within the loop dictionary, the data items
        are stored in key: value pairs as above but where the value is
        now a list of one or more data values assigned to that data
        name in the loop.

    Parameters
    ----------
    filepath: str
        Filepath to the input :term:`CIF` file.

    Returns
    -------
    dict[dict]
        A dictionary of the extracted data organised by :term:`data
        block`.

    Raises
    ------
    CIFParseError:
        If a syntax error is found in the input raw :term:`CIF` data.

    Examples
    --------


    """
    if not filepath.lower().endswith('.cif'):
        warnings.warn(("No .cif file extension detected. Assuming the filetype"
                       "is CIF and continuing."), UserWarning)
    p = CIFParser(filepath)
    p.parse()
    return dict((data_block.header, data_block.data_items)
                for data_block in p.data_blocks)


def validate_cif(filepath):
    """  Validate :term:`CIF` file syntax

    CIF file is scanned and checked for syntax errors. If one is found
    the error is reported explicitly, along with line number where it
    occurs. Returns ``True`` if no errors are found.


    Parameters
    ----------
    filepath: str
        Filepath to the input :term:`CIF` file.

    Returns
    -------
    bool
         Return ``True`` if file syntax is valid

    Raises
    ------
    CIFParseError:
        If a syntax error is found in the input raw :term:`CIF` data.

    Notes
    -----
    Syntax errors supported are:
        * Empty file
        * Missing inline :term:`data name`
        * Missing inline :term:`data value`
        * Unmatched loop :term:`data names` and :term:`data values`
        * Unclosed semicolon :term:`semicolon text field`

    Examples
    --------
    >>> validate_cif("path/to/valid_cif_file.cif")
    True
    >>> validate_cif("path/to/invalid_cif_file.cif")
    CIFParseError: Missing inline data name on line 3: "some_lone_data_value"
    """
    if not filepath.lower().endswith('.cif'):
        warnings.warn(("No .cif file extension detected. Assuming the filetype"
                       "is CIF and continuing."), UserWarning)
    with open(filepath, "r") as cif_file:
        raw_data = cif_file.read()
    v = CIFValidator(raw_data)
    return v.validate()


def cif2json(cif_filepath, json_filepath):
    """Save data in JSON format with data items sorted
        alphabetically by name.

        Parameters
        ----------
        filepath (str):
            Target filepath for output JSON file.
        """
    data = load_cif(cif_filepath)
    json_data = OrderedDict()
    for data_block_header, data_block in sorted(data.items()):
        json_data[data_block_header] = OrderedDict(sorted(data_block.items()))
    with open(json_filepath, 'w') as json_file:
        json_file.write(json.dumps(json_data, indent=4))


# Regular expressions used for parsing.
COMMENT_OR_BLANK = re.compile("\w*#.*|\s+$|^$")
DATA_BLOCK_HEADING = re.compile("(?:^|\n)(data_\S*)\s*", re.IGNORECASE)
LOOP = re.compile("(?:^|\n)loop_\s*", re.IGNORECASE)
DATA_NAME = re.compile("\s*_(\S+)")
DATA_NAME_START_LINE = re.compile("(?:^|\n)\s*_(\S+)")
DATA_VALUE = re.compile("\s*(\'[^\']+\'|\"[^\"]+\"|[^\s_#][^\s\'\"]*)")
TEXT_FIELD = re.compile("[^_][^;]+")
SEMICOLON_DATA_ITEM = re.compile(
    "(?:^|\n){0.pattern}\n;\n((?:.(?<!\n;))*)\n;".format(DATA_NAME), re.DOTALL)
INLINE_DATA_ITEM = re.compile(
    "(?:^|\n){0.pattern}[^\S\n]+{1.pattern}".format(DATA_NAME, DATA_VALUE))


class DataBlock:
    """Object for storing and extracting data for given
    :term:`data block`

    Parameters
    ----------
        header: str
            The :term:`data block header` of the data block
        raw_data: str
            The raw data from the :term:`data block` not including the
            :term:`data block header`, with the comments and blank
            lines stripped out.
        data_items: dict
            An empty dictionary

    Attributes
    ----------
        header: str
            The :term:`data block header` of the data block
        raw_data: str
            The raw data from the :term:`data block` from which the
            :term:`data items` are extracted. Data items are stripped
            out after extraction.
        data_items: dict
            A dictionary in which the :term:`data items` are stored as
            :term:`data name`: :term:`data value` pairs.
    """
    def __init__(self, header, raw_data, data_items):
        self.header = header
        self.raw_data = raw_data
        self.data_items = data_items

    def extract_data_items(self, data_item_pattern):
        """Extract matching (non-:term:`loop`) :term:`data items`

        Data items matching input `pattern` are extracted from
        `raw_data` and saved in the `data_item` dictionary. The
        matching data items are then removed from the `raw_data`
        string.

        Parameters
        ----------
        data_item_pattern:
            The compiled regex pattern which matches the
            :term:`data items` to be extracted. `pattern` must capture
            the :term:`data name` and :term:`data value`.

        Notes
        -----
        Only used for inline and semicolon :term:`data items`. However,
        any valid `data_item_pattern` should work.

        """
        data_items = data_item_pattern.findall(self.raw_data)
        for data_name, data_value in data_items:
            if data_item_pattern is SEMICOLON_DATA_ITEM:
                data_value = "'{}'".format(data_value)
            self.data_items[data_name] = data_value
        self.raw_data = data_item_pattern.sub("", self.raw_data)

    def extract_loop_data_items(self):
        """Extract all :term:`loop` :term:`data items` from raw data.

        The data items are extracted and stored in a dictionary where
        the keys are the :term:`data names` and the values are lists
        of the corresponding :term:'data values' i.e.

          .. code-block:: python

            {"data_name_A": ["data_value_A1", "data_value_A2", ...],
             "data_name_B": ["data_value_B1", "data_value_B2", ...]}

        The loop is then added to the `DataBlock` `data_items`
        dictionary with the key `"loop_i"` where i is the number of
        the loop in the order they appear in the :term:`CIF`.
        """
        loops = LOOP.split(self.raw_data)[1:]
        for i, loop in enumerate(loops):
            data_names = DATA_NAME_START_LINE.findall(loop)
            loop_data_items = {data_name: [] for data_name in data_names}
            data_value_lines = loop.split("\n")[len(data_names):]
            for line in data_value_lines:
                data_values = DATA_VALUE.findall(line)
                for data_name, data_value in zip(data_names, data_values):
                    loop_data_items[data_name].append(data_value)
            self.data_items["loop_{}".format(i + 1)] = loop_data_items

    def __repr__(self):
        """Representation of DataBlock, abbreviating raw data"""
        if len(self.raw_data) > 18:
            raw_data = "{:.15s}...".format(self.raw_data)
        else:
            raw_data = self.raw_data
        return "DataBlock({!r}, {!r}, {!r})".format(
            self.header, raw_data, self.data_items)

    def __eq__(self, other):
        return (self.header == other.header and
                self.data_items == other.data_items)


class CIFParser:
    """Class interface for parsing :term:`CIF` files and exporting data

    The CIF file is parsed and :term:`data items` are extracted for
    each :term:`data block`. Data is stored in a list of
    :class:`DataBlock` objects. This can then be saved to a JSON file.

    Before parsing the CIF file is checked and :class:`CIFParseError`
    is raised if a syntax error is found.

    Parameters
    ----------
    filepath: str
        Filepath to the input CIF file.

    Attributes
    ----------
    raw_data: str
        The raw data from the file of which the comments and blank
        lines are stripped out.
    data_blocks: list[DataBlock]
        A list of data blocks.

    Examples
    --------
    >>> p = CIFParser("path/to/cif_file.cif")
    >>> p.parse()
    >>> p.save("path/to/json_file.cif")
    """

    def __init__(self, filepath):
        with open(filepath, "r") as cif_file:
            self.raw_data = cif_file.read()
        validator = CIFValidator(self.raw_data)
        validator.validate()
        self.data_blocks = []

    def _strip_comments_and_blank_lines(self):
        """Remove all comments and blank lines raw file string."""
        lines = self.raw_data.split("\n")
        keep = [line for line in lines if not COMMENT_OR_BLANK.match(line)]
        self.raw_data = "\n".join(keep)

    def _extract_data_blocks(self):
        """Split raw file string into data blocks and save as a list
        of :class:`DataBlock` objects.
        """
        self.data_blocks = []
        data_blocks = DATA_BLOCK_HEADING.split(self.raw_data)[1:]
        headers, blocks = data_blocks[::2], data_blocks[1::2]
        for header, data in zip(headers, blocks):
            self.data_blocks.append(DataBlock(header, data, {}))

    def parse(self):
        """Parse the :term:`CIF` file by :term:`data block` and
        extract the :term:`data items`.2

        File is split into data blocks, each one saved in a
        :class:`DataBlock` object. Then for each data block, extract
        the semicolon, inline and loop data items.

        Notes
        -----
        When a data item is extracted, the corresponding raw CIF data
        is removed from the ``DataBlock.raw_data`` attribute and the
        extraction methods use this fact. Therefore, the data items
        must be extracted in the above order.

        For example, the ``DataBlock.extract_loop_data_items`` method
        assumes that only :term:`loop` data remains in the raw data
        and hence this must be done last.
        """
        self._strip_comments_and_blank_lines()
        self._extract_data_blocks()
        for data_block in self.data_blocks:
            data_block.extract_data_items(SEMICOLON_DATA_ITEM)
            data_block.extract_data_items(INLINE_DATA_ITEM)
            data_block.extract_loop_data_items()



class CIFParseError(Exception):
    """Exception for all parse errors due to incorrect syntax."""


class CIFValidator:
    """Class interface for validating CIF file syntax

    CIF file is scanned and checked for syntax errors. If one is found
    the error is reported explicitly, along with line number where it
    occurs.

    Parameters
    ----------
        raw_data: str
            The raw CIF file contents

    Attributes
    ----------
    lines: generator
        ``generator`` consisting of lines of the input CIF file
    current_line: str
        The current line being validated
    line_number: int
        The line number of the `current_line`

    Notes
    -----
    Current syntax errors supported are:
        * Empty file
        * Missing inline :term:`data name`
        * Missing inline :term:`data value`
        * Unmatched loop :term:`data names` and :term:`data values`
        * Unclosed semicolon :term:`semicolon text field`
    """
    def __init__(self, raw_data):
        """Initialises the :class:`CIFValidator` instance.

        The raw data of the CIF file is split by the newline character
        and stored in a generator. The :class:`CIFValidator` instance
        is initialised on the first line, warning the user a if the
        file is empty.
        """
        if not raw_data or raw_data.isspace():
            warnings.warn("File is empty.")
        self.lines = (line for line in raw_data.split("\n"))
        self.current_line = next(self.lines)
        self.line_number = 1

    def error(self, message=None, line_number=None, line=None):
        """Raise error message reporting the line number and line contents."""
        if line_number is None:
            line_number, line = self.line_number, self.current_line
        raise CIFParseError('{} on line {}: "{}"'.format(
            message, line_number, line))

    def validate(self):
        """Validate the :term:`CIF` file line by line

        Perform context sensitive, line by line, scan through CIF file
        checking the syntax is valid. Current contexts treated are top
        level, inside a :term:`loop` and inside a :term:`semicolon
        text field`.

        Returns
        -------
        bool
            Return ``True`` if file syntax is valid.

        Raises
        ------
        CIFParserError
            When a syntax error is found in the input raw CIF data.

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

    def _next_line(self):
        """Load next line of file and increment the current line number."""
        self.current_line = next(self.lines)
        self.line_number += 1

    def _validate_loop(self):
        """Validate :term:`loop` syntax.

        Raise a :class:`CIFParseError` if, for any row in the loop,
        the number of :term:`data values` does not match the number of
        declared :term:`data names`.

        Raises
        ------
        CIFParserError
            If number of :term:`data values` does not match the number
            of declared :term:`data names`.

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

    def _get_loop_data_names(self):
        """ Extract :term:`data names` from a :term:`loop`

        Collect and return a the list of data names declared at the
        beginning of a loop. The first line containing anything but a
        valid data name, comment or blank line will signify the end of
        the data names and is assumed to be the beginning of the loop
        :term:`data values`.

        Returns
        -------
        loop_data_names
            ``list`` of :term:`data names` in :term:`loop`.
        """
        loop_data_names = []
        self._next_line()
        while True:
            if COMMENT_OR_BLANK.match(self.current_line):
                self._next_line()
            elif DATA_NAME.match(self.current_line):
                loop_data_names.append(DATA_NAME.match(self.current_line))
                self._next_line()
            else:
                break
        return loop_data_names

    def _validate_lone_data_name(self):
        """Validate isolated :term:`data name`.

        An isolated :term:`data name` could indicate a missing an
        inline :term:`data value`, in which case raise a
        :class:`CIFParseError`. Otherwise it denotes the beginning of
        a :term:`semicolon data item`, in which case that validate
        that separately.

        Raises
        ------
        CIFParseError:
            If :term:`data name` is not matched with corresponding
            :term:`data value`.
        """
        err_line_number, err_line = self.line_number, self.current_line
        try:
            self._next_line()
        # check if final line of file
        except StopIteration:
            self.error("Invalid inline data value",
                       err_line_number, err_line)
        # check if part of semicolon data item
        if self.current_line.startswith(";"):
            self._validate_semicolon_data_item()
        else:
            self.error("Invalid inline data value",
                       err_line_number, err_line)

    def _validate_semicolon_data_item(self):
        """Validates :term:`semicolon data item`.

        Check for closing semicolon and raise a :class:`CIFParseError`
        if the :term:`semicolon text field` is left unclosed.

        Raises
        ------
        CIFParseError:
            If the :term:`semicolon text field` has no closing ``;``.
        """
        self._next_line()
        # two line queue must be kept as if no closing semicolon is found,
        # then error occurred on previous line.
        previous_lines = deque(maxlen=2)
        while True:
            if (COMMENT_OR_BLANK.match(self.current_line) or
                    TEXT_FIELD.match(self.current_line)):
                previous_lines.append((self.line_number, self.current_line))
                try:
                    self._next_line()
                # check if final line of file
                except StopIteration:
                    self.error("Unclosed semicolon text field")
            else:
                break
        if not self.current_line.startswith(";"):
            self.error("Unclosed semicolon text field",
                       *previous_lines[1])
        self._next_line()

    def _is_valid_single_line(self):
        """Check if valid single line (in top level context)

        Check the line is valid and necessitates no further validation
        of current or following lines. (Top level context meaning not
        inside a :term:`loop` or :term:`semicolon text field`.)
        """
        return (COMMENT_OR_BLANK.match(self.current_line) or
                INLINE_DATA_ITEM.match(self.current_line) or
                DATA_BLOCK_HEADING.match(self.current_line))

    def _is_loop_data_values(self):
        """Check if valid :term:`data value` in a :term:`loop` context."""
        return (DATA_VALUE.match(self.current_line) and not
                LOOP.match(self.current_line) and not
                DATA_BLOCK_HEADING.match(self.current_line))
