import re
import warnings


def load_cif(filepath):
    if not filepath.lower().endswith('.cif'):
        warnings.warn(("No .cif file extension detected. Assuming the filetype"
                       "is CIF and continuing."), UserWarning)
    with open(filepath, "r") as f:
        pass


COMMENT_OR_BLANK = re.compile("#.*|\s+|^$")
INLINE_NAME_VALUE = re.compile("_(\S+)\s+([\S|\' \']+)")
LOOP = re.compile("loop_\s", re.IGNORECASE)
DATA_NAME = re.compile("_(\S+)")
DATA_VALUE = re.compile("(\'[^\']+\'|\"[^\"]+\"|[^\s\'\"]+)")


class CIFParser:
    def __init__(self, filepath):
        with open(filepath, "r") as cif_file:
            self.raw_data = cif_file.read()
        self.data_items = {}

    def strip_comments_and_blank_lines(self):
        lines = self.raw_data.split("\n")
        keep = [line for line in lines if not COMMENT_OR_BLANK.match(line)]
        self.raw_data = '\n'.join(keep)

    def extract_inline_data_items(self):
        lines = self.raw_data.split("\n")
        keep = []
        for line in lines:
            matches = INLINE_NAME_VALUE.match(line)
            if matches:
                data_name, data_value = matches.groups()
                self.data_items[data_name] = data_value
            else:
                keep.append(line)
        self.raw_data = "\n".join(keep)

    def extract_loop_data_items(self):
        loops = LOOP.split(self.raw_data)[1:]
        for loop in loops:
            data_names = DATA_NAME.findall(loop)
            self.data_items.update({data_name: [] for data_name in data_names})
            data_value_lines = loop.split("\n")[len(data_names):]
            for line in data_value_lines:
                data_values = [m.group() for m in DATA_VALUE.finditer(line)]
                for data_name, data_value in zip(data_names, data_values):
                    self.data_items[data_name].append(data_value)
