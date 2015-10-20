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


COMMENT_OR_BLANK = re.compile(r"#.*|\s+$|^$")
SEMICOLON_FIELD = re.compile(r"(?:\n|^)_(\S+)\n;\n([^;]+)\n;")
DATA_BLOCK_HEADING = re.compile(r"(?:\n|^)(data_\S*)\s*\n", re.IGNORECASE)
INLINE_NAME_VALUE = re.compile("_(\S+)\s+([\S|\' ]+)")
LOOP = re.compile(r"(?:\n|^)loop_\s*\n", re.IGNORECASE)
DATA_NAME = re.compile("_(\S+)")
DATA_VALUE = re.compile("(\'[^\']+\'|\"[^\"]+\"|[^\s\'\"]+)")

# mutable data structure for saving components of data blocks
DataBlock = recordclass("DataBlock", "heading raw_data data_items")


class CIFParser:
    def __init__(self, filepath):
        with open(filepath, "r") as cif_file:
            self.raw_data = cif_file.read()
        self.data_blocks = []

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
    def extract_semicolon_data_items(data_block):
        data_items = SEMICOLON_FIELD.findall(data_block.raw_data)
        for data_name, data_value in data_items:
            data_block.data_items[data_name] = "'{}'".format(data_value)
        data_block.raw_data = SEMICOLON_FIELD.sub("", data_block.raw_data)

    @staticmethod
    def extract_inline_data_items(data_block):
        lines = data_block.raw_data.split("\n")
        keep = []
        for line in lines:
            matches = INLINE_NAME_VALUE.match(line)
            if matches:
                data_name, data_value = matches.groups()
                data_block.data_items[data_name] = data_value
            else:
                keep.append(line)
        data_block.raw_data = "\n".join(keep)

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
            self.extract_semicolon_data_items(data_block)
            self.extract_inline_data_items(data_block)
            self.extract_loop_data_items(data_block)

    def save(self, filepath):
        with open(filepath, 'w') as f:
            json_data = OrderedDict()
            json_data.keys()
            for data_block in self.data_blocks:
                json_data[data_block.heading] = OrderedDict(
                    sorted(data_block.data_items.items()))
            f.write(json.dumps(json_data, indent=4))
