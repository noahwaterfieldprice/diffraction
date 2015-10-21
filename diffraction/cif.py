import json
import re
import warnings
from recordclass import recordclass
from collections import OrderedDict


def load_cif(filepath):
    if not filepath.lower().endswith('.cif'):
        warnings.warn(("No .cif file extension detected. Assuming the filetype"
                       "is CIF and continuing."), UserWarning)
    with open(filepath, "r") as f:
        pass


LINE = re.compile("([^\n]+)")
COMMENT_OR_BLANK = re.compile("#.*|\s+$|^$")
DATA_BLOCK_HEADING = re.compile("(?:\n|^)(data_\S*)\s*\n", re.IGNORECASE)
LOOP = re.compile("(?:\n|^)loop_\s*\n", re.IGNORECASE)
DATA_NAME = re.compile("_(\S+)")
DATA_VALUE = re.compile("(\'[^\']+\'|\"[^\"]+\"|[^\s\'\"]+)")
SEMICOLON_DATA_ITEM = re.compile(r"(?:\n|^)_(\S+)\n;\n([^;]+)\n;")
INLINE_DATA_ITEM = re.compile("(?:\n|^)" + DATA_NAME.pattern +
                              r"[^\S\n]+" + DATA_VALUE.pattern)


# mutable data structure for saving components of data blocks
DataBlock = recordclass("DataBlock", "heading raw_data data_items")


class CIFParseError(Exception):
    pass


class CIFParser:
    def __init__(self, filepath):
        with open(filepath, "r") as cif_file:
            self.raw_data = cif_file.read()
        self.data_blocks = []

    def error(self, message):
        raise CIFParseError(message)

    def validate(self):
        lines = (line.group(0) for line in LINE.finditer(self.raw_data))
        while True:
            line = next(lines)
            print(line)
            if COMMENT_OR_BLANK.match(line):
                continue
            if LOOP.match(line):
                print("LOOOOOOOOP")

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
            data_names = DATA_NAME.findall(loop)
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
