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
DataBlock = recordclass("DataBlock", "heading raw_data data_items")


class CIFParser:
    def __init__(self, filepath):
        with open(filepath, "r") as cif_file:
            self.raw_data = cif_file.read()
        self.data_blocks = []

    def validate(self):
        validator = CIFValidator(self.raw_data)
        try:
            validator.validate()
        except StopIteration:
            print('valid file')

    def strip_comments_and_blank_lines(self):
        lines = self.raw_data.split("\n")
        keep = [line for line in lines if not COMMENT_OR_BLANK.match(line)]
        self.raw_data = "\n".join(keep)

    def extract_data_blocks(self):
        data_blocks = DATA_BLOCK_HEADING.split(self.raw_data)[1:]
        headings, blocks = data_blocks[::2], data_blocks[1::2]
        for heading, data in zip(headings, blocks):
            self.data_blocks.append(DataBlock(heading, data, {}))

    @staticmethod
    def extract_data_items(data_block, pattern):
        data_items = pattern.findall(data_block.raw_data)
        for data_name, data_value in data_items:
            if pattern is SEMICOLON_DATA_ITEM:
                data_value = "'{}'".format(data_value)
            data_block.data_items[data_name] = data_value
        data_block.raw_data = pattern.sub("", data_block.raw_data)

    @staticmethod
    def extract_loop_data_items(data_block):
        loops = LOOP.split(data_block.raw_data)[1:]
        for i, loop in enumerate(loops):
            data_names = SL_DATA_NAME.findall(loop)
            loop_data_items = {data_name: [] for data_name in data_names}
            data_value_lines = loop.split("\n")[len(data_names):]
            for line in data_value_lines:
                data_values = DATA_VALUE.findall(line)
                for data_name, data_value in zip(data_names, data_values):
                    loop_data_items[data_name].append(data_value)
            data_block.data_items["loop_{}".format(i + 1)] = loop_data_items

    def parse(self):
        self.strip_comments_and_blank_lines()
        self.extract_data_blocks()
        for data_block in self.data_blocks:
            self.extract_data_items(data_block, SEMICOLON_DATA_ITEM)
            self.extract_data_items(data_block, INLINE_DATA_ITEM)
            self.extract_loop_data_items(data_block)

    def save(self, filepath):
        with open(filepath, 'w') as f:
            json_data = OrderedDict()
            json_data.keys()
            for data_block in self.data_blocks:
                json_data[data_block.heading] = OrderedDict(
                    sorted(data_block.data_items.items()))
            f.write(json.dumps(json_data, indent=4))


class CIFParseError(Exception):
    """Exception raised for all parse errors."""


class CIFValidator:
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

    def next_line(self):
        self.current_line = next(self.lines)
        self.line_number += 1

    def validate(self):
        while True:
            if (COMMENT_OR_BLANK.match(self.current_line) or
                    INLINE_DATA_ITEM.match(self.current_line) or
                    DATA_BLOCK_HEADING.match(self.current_line)):
                self.next_line()
            elif LOOP.match(self.current_line):
                self.validate_loop()
            elif DATA_VALUE.match(self.current_line.lstrip()):
                self.error("Missing inline data name")
            elif DATA_NAME.match(self.current_line):
                err_line_number, err_line = self.line_number, self.current_line
                try:
                    self.next_line()
                # check if final line of file
                except StopIteration:
                    self.error("Invalid inline data value",
                               err_line_number, err_line)
                # check if part of semicolon data item
                if self.current_line.startswith(";"):
                    self.validate_semicolon_data_item()
                else:
                    self.error("Invalid inline data value",
                               err_line_number, err_line)
            else:
                print(self.line_number, ':'*10, self.current_line)

    def validate_loop(self):
        loop_data_names = 0
        self.next_line()
        while True:
            if COMMENT_OR_BLANK.match(self.current_line):
                self.next_line()
            elif DATA_NAME.match(self.current_line):
                loop_data_names += 1
                self.next_line()
            else:
                break
        while True:
            if COMMENT_OR_BLANK.match(self.current_line):
                self.next_line()
            elif (DATA_VALUE.match(self.current_line) and not
                    LOOP.match(self.current_line) and not
                    DATA_BLOCK_HEADING.match(self.current_line)):
                data_values = DATA_VALUE.findall(self.current_line)
                if len(data_values) != loop_data_names:
                    self.error("Unmatched data values to data names in loop")
                self.next_line()
            else:
                break

    def validate_semicolon_data_item(self):
        self.next_line()
        previous_lines = deque(maxlen=2)
        while TEXT_FIELD.match(self.current_line):
            previous_lines.append((self.line_number, self.current_line))
            try:
                self.next_line()
            # check if final line of file
            except StopIteration:
                self.error("Unclosed semicolon text field")
        if not self.current_line.startswith(";"):
            print(previous_lines)
            self.error("Unclosed semicolon text field",
                       *previous_lines[0])
        self.next_line()
