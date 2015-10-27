import json
import re
import warnings
from collections import OrderedDict, deque
from itertools import count
from recordclass import recordclass


def load_cif(filepath):
    if not filepath.lower().endswith('.cif'):
        warnings.warn(("No .cif file extension detected. Assuming the filetype"
                       "is CIF and continuing."), UserWarning)
    with open(filepath, "r") as f:
        pass


# Regular expressions used for parsing.
COMMENT_OR_BLANK = re.compile("#.*|\s+$|^$")
DATA_BLOCK_HEADING = re.compile(r"(?:^|\n)(data_\S*)\s*", re.IGNORECASE)
LOOP = re.compile(r"(?:^|\n)loop_\s*", re.IGNORECASE)
DATA_NAME = re.compile(r"\s*_(\S+)")
SL_DATA_NAME = re.compile(r"(?:^|\n)\s*_(\S+)")
DATA_VALUE = re.compile(r"\s*(\'[^\']+\'|\"[^\"]+\"|[^\s_#][^\s\'\"]*)")
TEXT_FIELD = re.compile(r"[^_][^;]+")
SEMICOLON_DATA_ITEM = re.compile(
    "(?:^|\n)" + DATA_NAME.pattern + "\n;\n([^;]+)\n;")
INLINE_DATA_ITEM = re.compile(
    "(?:^|\n)" + DATA_NAME.pattern + r"[^\S\n]+" + DATA_VALUE.pattern)

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
    def __init__(self, filepath):
        with open(filepath, "r") as cif_file:
            self.raw_data = cif_file.read()
        self.data_blocks = []

    def _validate_syntax(self):
        validator = CIFSyntaxValidator(self.raw_data)
        try:
            validator.validate()
        except StopIteration:
            print('valid file')

    def _strip_comments_and_blank_lines(self):
        lines = self.raw_data.split("\n")
        keep = [line for line in lines if not COMMENT_OR_BLANK.match(line)]
        self.raw_data = "\n".join(keep)

    def _extract_data_blocks(self):
        self.data_blocks = []
        data_blocks = DATA_BLOCK_HEADING.split(self.raw_data)[1:]
        headings, blocks = data_blocks[::2], data_blocks[1::2]
        for heading, data in zip(headings, blocks):
            self.data_blocks.append(DataBlock(heading, data, {}))

    def parse(self):
        self._strip_comments_and_blank_lines()
        self._extract_data_blocks()
        for data_block in self.data_blocks:
            data_block.extract_data_items(SEMICOLON_DATA_ITEM)
            data_block.extract_data_items(INLINE_DATA_ITEM)
            data_block.extract_loop_data_items()

    def save(self, filepath):
        with open(filepath, 'w') as f:
            json_data = OrderedDict()
            json_data.keys()
            for data_block in self.data_blocks:
                json_data[data_block.heading] = OrderedDict(
                    sorted(data_block.data_items.items()))
            f.write(json.dumps(json_data, indent=4))


class CIFParseError(Exception):
    """Exception raised for all parse errors due to incorrect syntax."""


class CIFSyntaxValidator:
    def __init__(self, raw_data):
        self.raw_data = raw_data
        self.lines = (line for line in raw_data.split("\n"))
        # initialise on first line
        self.line_number = 1
        self.current_line = next(self.lines)

    def error(self, message=None, line_number=None, line=None):
        if line_number is None:
            line_number, line = self.line_number, self.current_line
        raise CIFParseError('{} on line {}: "{}"'.format(
            message, line_number, line))

    def validate(self):
        while True:
            if self._is_valid_inline_data_item():
                self._next_line()
            elif LOOP.match(self.current_line):
                self._validate_loop()
            elif DATA_VALUE.match(self.current_line.lstrip()):
                self.error("Missing inline data name")
            elif DATA_NAME.match(self.current_line):
                self._validate_lone_data_name()

    def _next_line(self):
        self.current_line = next(self.lines)
        self.line_number += 1

    def _validate_loop(self):
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
        self._next_line()
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

    def _is_valid_inline_data_item(self):
        return (COMMENT_OR_BLANK.match(self.current_line) or
                INLINE_DATA_ITEM.match(self.current_line) or
                DATA_BLOCK_HEADING.match(self.current_line))

    def _is_loop_data_values(self):
        return (DATA_VALUE.match(self.current_line) and not
        LOOP.match(self.current_line) and not
                DATA_BLOCK_HEADING.match(self.current_line))


class CIFInterpreter:
    LOOP_NAMES = (
        ("symmetry_equiv_pos", "symmetry_equiv_pos"),
        ("atom_site_fract", "atom_sites"),
        ("atom_site_aniso", "atom_site_aniso"),
        ("atom_type_oxidation", "atom_oxidations"),
        ("publ_author_name", "publ_author_names"),
        ("citation", "citation"),
        ("atom_type_radius_bond", "atom_bonds")
    )

    def __init__(self, data_blocks):
        self.data_blocks = data_blocks

    def rename_loops(self):
        for data_block in self.data_blocks:
            for old_name in ("loop_{}".format(i + 1) for i in count(0)):
                try:
                    self.rename_loop(old_name, data_block)
                except KeyError:
                    break

    def rename_loop(self, old_name, data_block):
        loop = data_block.data_items[old_name]
        for key_data_name, loop_name in self.LOOP_NAMES:
            if any(re.match(key_data_name, data_name)
                   for data_name in loop.keys()):
                data_block.data_items[loop_name] = loop
                data_block.data_items.pop(old_name)
