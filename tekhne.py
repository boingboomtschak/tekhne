#! /usr/bin/python3
import argparse, os
from lark import Lark

parser = argparse.ArgumentParser(description="CUDA to WGSL transpiler")
parser.add_argument("input", help="Path to input .cu file")
parser.add_argument("-o", "--output", help="Path to output .wgsl file")
args = vars(parser.parse_args())

input_path = args['input']
output_path = args['output'] if args['output'] else os.path.splitext(args.get("input"))[0] + '.wgsl'

l = Lark('''
start: WORD "," WORD "!"
%import common.WORD
%ignore " "
''')