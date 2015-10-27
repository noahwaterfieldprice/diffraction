"""
Docstring for the module
"""


import json
import re
import warnings
from collections import OrderedDict, deque
from recordclass import recordclass


def load_cif(filepath):
    if not filepath.lower().endswith('.cif'):
        warnings.warn(("No .cif file extension detected. Assuming the filetype"
                       "is CIF and continuing."), UserWarning)
    with open(filepath, "r") as f:
        p = CIFParser(filepath)
        p.parse()
    if len(p) == 1:
        return p.data_blocks[0].data_items
    return p.data_blocks

# Regular expressions used for parsing.
COMMENT_OR_BLANK = re.compile("\w*#.*|\s+$|^$")
DATA_BLOCK_HEADING = re.compile(r"(?:^|\n)(data_\S*)\s*", re.IGNORECASE)
LOOP = re.compile(r"(?:^|\n)loop_\s*", re.IGNORECASE)
DATA_NAME = re.compile(r"\s*_(\S+)")
SL_DATA_NAME = re.compile(r"(?:^|\n)\s*_(\S+)")
DATA_VALUE = re.compile(r"\s*(\'[^\']+\'|\"[^\"]+\"|[^\s_#][^\s\'\"]*)")
TEXT_FIELD = re.compile(r"[^_][^;]+")
SEMICOLON_DATA_ITEM = re.compile(
    r"(?:^|\n)" + DATA_NAME.pattern + r"\n;\n([^;]+)\n;")
INLINE_DATA_ITEM = re.compile(
    r"(?:^|\n)" + DATA_NAME.pattern + "[^\\S\\n]+" + DATA_VALUE.pattern)

# Mutable data structure for saving components of data blocks.
DataBlockRecord = recordclass("DataBlock", "heading raw_data data_items")


class DataBlock(DataBlockRecord):
    def __init__(self, heading, raw_data, data_items):
        super().__init__(heading, raw_data, data_items)

    def extract_data_items(self, pattern):
        data_items = pattern.findall(self.raw_data)
        for data_name, data_value in data_items:
            if pattern is SEMICOLON_DATA_ITEM:
                data_value = "'{}'".format(data_value)
            self.data_items[data_name] = data_value
        self.raw_data = pattern.sub("", self.raw_data)

    def extract_loop_data_items(self):
        loops = LOOP.split(self.raw_data)[1:]
        for i, loop in enumerate(loops):
            data_names = SL_DATA_NAME.findall(loop)
            loop_data_items = {data_name: [] for data_name in data_names}
            data_value_lines = loop.split("\n")[len(data_names):]
            for line in data_value_lines:
                data_values = DATA_VALUE.findall(line)
                for data_name, data_value in zip(data_names, data_values):
                    loop_data_items[data_name].append(data_value)
            self.data_items["loop_{}".format(i + 1)] = loop_data_items


class CIFParser:
    """ Class for parsing CIF files and exporting data in JSON style format

    The CIF file is parsed and data items are extracted for each data block.
    Data is stored in a list of :class:`DataBlock` objects. This can then be
    saved to a .json file.

    Before parsing the CIF file is checked for syntax errors and if one is
    found then a :class:`CIFParseError` is raised


    Attributes
    ----------
    raw_data: str
        A string of the raw data from the file of which the comments and
        blank lines are stripped out
    data_blocks: list[DataBlock]
        A list of data blocks




    Examples
    --------
    >>> p = CIFParser("path/to/cif_file.cif")
    >>> p.parse()
    >>> p.save("path/to/json_file.cif")
    """

    def __init__(self, filepath):
        """
        Parameters
        ----------
        filepath: str

        """

        with open(filepath, "r") as cif_file:
            self.raw_data = cif_file.read()
        validator = CIFValidator(self.raw_data)
        validator.validate()
        self.data_blocks = []

    def _strip_comments_and_blank_lines(self):
        """
        Remove all comments and blank lines from stored raw file string.
        """
        lines = self.raw_data.split("\n")
        keep = [line for line in lines if not COMMENT_OR_BLANK.match(line)]
        self.raw_data = "\n".join(keep)

    def _extract_data_blocks(self):
        """
        Split raw file string into data blocks and
        save as a list of DataBlock objects.
        """
        self.data_blocks = []
        data_blocks = DATA_BLOCK_HEADING.split(self.raw_data)[1:]
        headings, blocks = data_blocks[::2], data_blocks[1::2]
        for heading, data in zip(headings, blocks):
            self.data_blocks.append(DataBlock(heading, data, {}))

    def parse(self):
        """


        """
        self._strip_comments_and_blank_lines()
        self._extract_data_blocks()
        for data_block in self.data_blocks:
            data_block.extract_data_items(SEMICOLON_DATA_ITEM)
            data_block.extract_data_items(INLINE_DATA_ITEM)
            data_block.extract_loop_data_items()

    def save(self, filepath):
        """
        Save data in JSON format with data items sorted alphabetically by name.

        Args:
            filepath (str): target filepath for json file
        """

        with open(filepath, 'w') as f:
            json_data = OrderedDict()
            json_data.keys()
            for data_block in self.data_blocks:
                json_data[data_block.heading] = OrderedDict(
                    sorted(data_block.data_items.items()))
            f.write(json.dumps(json_data, indent=4))


class CIFParseError(Exception):
    """Exception raised for all parse errors due to incorrect syntax."""


class CIFValidator:
    """

    Attributes
    ----------
    lines: generator
        ``generator`` consisting of lines of the input CIF file
    current_line: str
        The current line being validated
    line_number: int
        The line number of the `current_line`

    Raises
    ------
    CIFParserError
        When a syntax error is found in the input raw CIF data
    """
    def __init__(self, raw_data):
        """
        Init the :class:`CIFValidator` instance.

        The raw data of the CIF file is split by the ``\n`` newline character
        and stored in a generator. The :class:`CIFValidator` instance is
        initialised on the first line, raising a :class:`CIFParseError` if the
        file is empty.

        Parameters
        ----------
        raw_data: str

        """
        self.lines = (line for line in raw_data.split("\n"))
        try:
            self.current_line = next(self.lines)
            self.line_number = 1
        except StopIteration:
            raise CIFParseError("Empty file")

    def error(self, message=None, line_number=None, line=None):
        if line_number is None:
            line_number, line = self.line_number, self.current_line
        raise CIFParseError('{} on line {}: "{}"'.format(
            message, line_number, line))

    def validate(self):
        """
        docstring for validate
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
            pass

    def _next_line(self):
        self.current_line = next(self.lines)
        self.line_number += 1

    def _validate_loop(self):
        """Validate :term:`loop` syntax.

        Raise a :class:`CIFParseError` if, for any row in the :term:`loop`,
        the number of :term:`data value`s does not match the number of
        declared :term:`data name`s.
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
        """ Extract :term:`data name`s from a :term:`loop`

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

        An isolated :term:`data name` is could indicated a missing an inline
        :term:`data value`, in which case raise a :class:`CIFParseError`.
        Otherwise it denotes the beginning of a :term:`semicolon data item`,
        in which case that validate that separately.
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
        """Validates :term:`semicolon data item.`

        Check for closing semicolon and raise a :class:`CIFParseError` if a
        semicolon :term:`text field` is left unclosed.
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
        """Alias for check if line is valid and necessitates no further
        validation of current or following lines."""
        return (COMMENT_OR_BLANK.match(self.current_line) or
                INLINE_DATA_ITEM.match(self.current_line) or
                DATA_BLOCK_HEADING.match(self.current_line))

    def _is_loop_data_values(self):
        """Alias for check if valid :term:`loop` :term:`data value`."""
        return (DATA_VALUE.match(self.current_line) and not
                LOOP.match(self.current_line) and not
                DATA_BLOCK_HEADING.match(self.current_line))
