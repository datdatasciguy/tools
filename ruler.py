#!/usr/bin/env python
"""
ruler.py

A very simple utility to print a text file with a horizontal numeric ruler. Especially useful for
fixed-width text files.

Example output:
         10        20        30        40        50        60        70        80        90
123456789012345678901234567890123456789012345678901234567890123456789012345678901234567890123456789
Here is an example text file.
Above you can see the ruler.
Now you know the byte locations.

Usage:
    ruler.py <file> [--cadence N] [--length L]

Options:
    --cadence N   Insert the ruler every N lines (default: 20)
    --length L    Set custom ruler length (default: auto-detect)
"""

import argparse
import sys

description = __doc__

def parse_args():
    parser = argparse.ArgumentParser(
        description=description,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        'file',
        type=argparse.FileType('r'),
        help='Path to the input text file.'
    )
    parser.add_argument(
        '--cadence',
        dest='cadence',
        type=int,
        default=20,
        help='Insert the ruler every N lines (default: 20).'
    )
    parser.add_argument(
        '--length',
        dest='length',
        type=int,
        help='Custom ruler length (default: auto-detect).'
    )
    return parser.parse_args()

class Ruler:
    def __init__(self, lines, length=None, cadence=20):
        """
        :param lines:    List of input lines
        :param length:   Width of ruler; if None, use longest line
        :param cadence:  Print ruler at cadence N lines
        """
        self.lines = lines
        self.cadence = cadence
        self.length = length or self._detect_max_length()
        self._ruler_str = self._make_ruler()

    def _detect_max_length(self):
        return max(len(line.rstrip("\n")) for line in self.lines) if self.lines else 0

    def _make_ruler(self):
        """
        Two-line ruler
        - First line: multiples of 10
        - Second line: repeating 1234567890
        """
        marker_line = [" "] * self.length
        for pos in range(10, self.length + 1, 10):
            num_str = str(pos)
            zero_idx = pos - 1
            start_idx = zero_idx
            for i, ch in enumerate(num_str):
                idx = start_idx + i
                if idx < self.length:
                    marker_line[idx] = ch
        units_line = [str((i + 1) % 10) for i in range(self.length)]
        return "".join(marker_line) + "\n" + "".join(units_line)

    def print_with_ruler(self, out=sys.stdout):
        for idx, line in enumerate(self.lines, start=1):
            if (idx - 1) % self.cadence == 0:
                print(self._ruler_str, file=out)
            print(line.rstrip("\n"), file=out)

def main():
    args = parse_args()
    lines = args.file.readlines()
    ruler = Ruler(
        lines=lines,
        length=args.length,
        cadence=args.cadence
    )
    ruler.print_with_ruler()

if __name__ == "__main__":
    main()