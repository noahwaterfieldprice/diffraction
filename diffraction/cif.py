import re
import warnings


def load_cif(filepath):
    if not filepath.lower().endswith('.cif'):
        warnings.warn(("No .cif file extension detected. Assuming the filetype"
                       "is CIF and continuing."), UserWarning)
    with open(filepath, "r") as f:
        pass


COMMENT_OR_BLANK = re.compile("^#.*|\s+|^$")
INLINE_NAME_VALUE = re.compile("^_(\S+)\s+([\S|\' \']+)")


class CIFParser:
    def __init__(self, filepath):
        with open(filepath, "r") as cif_file:
            self.raw_data = cif_file.read()
        self.strip_comments_and_blank_lines()
        self.data = {}
        self.extract_data_name_data_value_pairs()

    def strip_comments_and_blank_lines(self):
        data = self.raw_data.split("\n")
        data = [line for line in data if not COMMENT_OR_BLANK.match(line)]
        self.raw_data = '\n'.join(data)

    def extract_data_name_data_value_pairs(self):
        data = self.raw_data.split("\n")
        for line in data:
            matches = INLINE_NAME_VALUE.match(line)
            if matches:
                data_name, data_value = matches.group(1, 2)
                self.data[data_name] = data_value
