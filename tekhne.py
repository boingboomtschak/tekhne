#! /usr/bin/env python3
'''Simple CUDA -> WGSL transpiler'''
import argparse
import os
import logging
import sys
import coloredlogs
from colorama import Fore, Style
from lark import Lark, Visitor
from lark import tree as LarkTree

# Setting up argument parser
argparser = argparse.ArgumentParser(description="CUDA to WGSL transpiler")
argparser.add_argument("input", help="Path to input .cu file")
argparser.add_argument("-o", "--output", help="Path to output .wgsl file")
argparser.add_argument("-d", "--debug", action="store_true", help="Show debug information")
argparser.add_argument("-f" ,"--file-log", action="store_true", help="Store logs to 'tekhne.log'")
argparser.add_argument("-t", "--parse-tree", action="store_true", \
    help="Save parse tree to 'parse-tree.png'")
args = vars(argparser.parse_args())

# Setting up logger
log = logging.getLogger(__name__)
streamFmt=f'{Fore.CYAN}[{Fore.GREEN}tekhne{Fore.CYAN}]{Style.RESET_ALL} %(message)s'
LOG_LEVEL = 'DEBUG' if args.get('debug') else 'INFO'
coloredlogs.install(fmt=streamFmt, level=LOG_LEVEL, logger=log)
if args.get("file_log"):
    log.debug("Logging to 'tekhne.log'...")
    FILE_FMT='[tekhne] %(asctime)s : %(message)s'
    fileHandler = logging.FileHandler('tekhne.log')
    fileFormatter = logging.Formatter(FILE_FMT)
    fileHandler.setFormatter(fileFormatter)
    log.addHandler(fileHandler)

input_path = os.path.abspath(args['input'])
input_fname = os.path.basename(args['input'])
output_path = args['output'] if args['output'] else os.path.splitext(input_fname)[0] + '.wgsl'

log.debug("Reading input file...")
CUDA = ''
try:
    with open(input_path, 'r', encoding='utf-8') as f:
        CUDA = f.read()
except FileNotFoundError:
    log.error('\'%s\' not found in \'%s\'!', input_fname, os.path.dirname(input_path))
    sys.exit(1)
except PermissionError:
    log.error('No permission to read \'%s\'!', input_path)
    sys.exit(1)
except OSError:
    log.error('OS error reading \'%s\'', input_path)
    sys.exit(1)

# Setting up CUDA grammar
CUDA_GRAMMAR = '''
%import common.WS
%import common.C_COMMENT
%import common.CPP_COMMENT
%ignore WS
%ignore C_COMMENT
%ignore CPP_COMMENT

%import common.CNAME
%import common.INT
%import common.SIGNED_INT
%import common.DECIMAL

BOOLEAN : "true"
        | "false"

?atom : INT
     | SIGNED_INT
     | DECIMAL
     | CNAME
     | BOOLEAN
     | "(" expression ")"
?level1 : atom ("++"|"--")?
       | level1 "(" (expression ("," expression)*)? ")"
       | level1 "[" expression "]"
       | level1 "." CNAME 
?level2 : ("++"|"+"|"--"|"-"|"!"|"*"|"~")? level1
?level3 : (level3 ("*"|"/"|"%"))? level2
?level4 : (level4 ("+"|"-"))? level3
?level5 : (level5 ("<<"|">>"))? level4
?level6 : (level6 ("<"|">"|"<="|">="))? level5
?level7 : (level7 ("=="|"!="))? level6
?level8 : (level8 "&")? level7
?level9 : (level9 "^")? level8
?level10 : (level10 "|")? level9
?level11 : (level11 "&&")? level10
?level12 : (level12 "||")? level11
expression : level12

lvalue : CNAME (("[" expression "]")|("." CNAME))*

assignment  : lvalue "=" expression ";"
            | lvalue "+=" expression ";"
            | lvalue "-=" expression ";"
            | lvalue "*=" expression ";"
            | lvalue "/=" expression ";"

ctype : CNAME 
ptrtype : CNAME "*"

cudaspec : ("__shared__"|"__global__"|"__device__")

declaration : cudaspec? ctype CNAME ("[" expression "]")* ("=" expression)? ";"
            | cudaspec? ctype CNAME ("," CNAME)* ";"

conditional : "if" "(" expression ")" (("{" statement* "}")|statement) ("else" ("{" statement* "}")|statement)*

while_loop : "while" "(" expression ")" (("{" statement* "}")|statement)

for_loop : "for" "(" (declaration|assignment) expression ";" expression ")" (("{" statement* "}")|statement)

statement : for_loop
          | while_loop
          | conditional
          | declaration
          | expression ";"
          | assignment

argument : (ctype|ptrtype) CNAME 

kerneldecl : CNAME "(" argument ("," argument)* ")" 

kernelspec : "__global__" ctype kerneldecl "{" statement* "}"

start : kernelspec*
'''

class WGSLCodeGenerator:        
    def visit(self, tree):
        print(tree.data)
        try:
            return getattr(self, tree.data)(tree)
        except AttributeError:
            return self.__default__(tree)
    def __default__(self, tree):
        log.info("Default visitor!")
    def start(self, tree):
        log.info("start")
        [self.visit(c) for c in tree.children]
    def kernelspec(self, tree):
        log.info("kernelspec")
        [self.visit(c) for c in tree.children]

# Setting up Lark parser
log.debug("Setting up parser...")
parser = Lark(CUDA_GRAMMAR, parser="lalr")

log.debug("Parsing input...")
parsed = parser.parse(CUDA)

if args['parse_tree']:
    log.debug("Generating parse tree to 'parse-tree.png'")
    LarkTree.pydot__tree_to_png(parsed, 'parse-tree.png')
    log.debug("Parse tree generated.")

log.debug("Running code generator...")
WGSLCodeGenerator().visit(parsed)

#log.debug(parsed)

log.debug("Exiting...")
