import re
import warnings


def load_cif(filepath):
    if not filepath.lower().endswith('.cif'):
        warnings.warn(("No .cif file extension detected. Assuming the filetype"
                      "is CIF and continuing."), UserWarning)
    with open(filepath, "r") as f:
        pass


class CIFParser:

    def __init__(self, filepath):
        with open(filepath, "r") as cif_file:
            self.raw_data = cif_file.read()
        self.strip_comments_and_blank_lines()

    def strip_comments_and_blank_lines(self):
        data = self.raw_data.split("\n")
        comment_or_blank = re.compile("^#.*|\s+|^$")
        data = [line for line in data if not comment_or_blank.match(line)]
        self.raw_data = '\n'.join(data)