#! /usr/bin/env python3
'''Simple CUDA -> WGSL transpiler'''
import argparse
import os
import logging
import sys
import coloredlogs
from colorama import Fore, Style
from lark import Lark, Token
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
?level2 : "++" level1 -> pre_inc
        | "+" level1 -> pos
        | "--" level1 -> pre_dec
        | "-" level1 -> neg
        | "!" level1 -> not
        | "*" level1 -> deref
        | "~" level1 
        | level1
?level3 : level3 "*" level2 -> mult
        | level3 "/" level2 -> div
        | level3 "%" level2 -> mod
        | level2
?level4 : level4 "+" level3 -> plus 
        | level4 "-" level3 -> minus
        | level3 
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

CUDASPEC : ("__shared__"|"__global__"|"__device__")

declaration : CUDASPEC? ctype CNAME ("[" expression "]")* ("=" expression)? ";"
            | CUDASPEC? ctype CNAME ("," CNAME)* ";"

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
        if isinstance(tree, Token):
            return str(tree)
        try:
            return getattr(self, tree.data)(tree)
        except AttributeError:
            return self.__default__(tree)
    def __default__(self, tree):
        log.warning(f"Default visitor visiting {tree.data}!")
        [self.visit(c) for c in tree.children]
    def start(self, tree):
        log.info("start")
        [self.visit(c) for c in tree.children]
    def kernelspec(self, tree):
        log.info("kernelspec")
        [self.visit(c) for c in tree.children]
    def kerneldecl(self, tree):
        log.info('kerneldecl')
        [self.visit(c) for c in tree.children]
    def argument(self, tree):
        log.info('argument')
        [self.visit(c) for c in tree.children]
    def statement(self, tree):
        log.info('statement')
        [self.visit(c) for c in tree.children]
    def for_loop(self, tree):
        log.info('for loop')
        [self.visit(c) for c in tree.children]
    def while_loop(self, tree):
        log.info('while loop')
        [self.visit(c) for c in tree.children]
    def conditional(self, tree):
        log.info('conditonal')
        [self.visit(c) for c in tree.children]
    def declaration(self, tree):
        log.info('declaration')
        [self.visit(c) for c in tree.children]
    def ctype(self, tree):
        log.info('ctype')
        [self.visit(c) for c in tree.children]
    def ptrtype(self, tree):
        log.info('ptrtype')
        [self.visit(c) for c in tree.children]
    def assignment(self, tree):
        log.info('assignment')
        [self.visit(c) for c in tree.children]
    def lvalue(self, tree):
        log.info('lvalue')
        [self.visit(c) for c in tree.children]
    def expression(self, tree):
        log.info('expression')
        [self.visit(c) for c in tree.children]
    def level12(self, tree):
        log.info('level12')
        [self.visit(c) for c in tree.children]
    def level11(self, tree):
        log.info('level11')
        [self.visit(c) for c in tree.children]
    def level10(self, tree):
        log.info('level10')
        [self.visit(c) for c in tree.children]
    def level9(self, tree):
        log.info('level9')
        [self.visit(c) for c in tree.children]
    def level8(self, tree):
        log.info('level8')
        [self.visit(c) for c in tree.children]
    def level7(self, tree):
        log.info('level7')
        [self.visit(c) for c in tree.children]
    def level6(self, tree):
        log.info('level6')
        [self.visit(c) for c in tree.children]
    def level5(self, tree):
        log.info('level5')
        [self.visit(c) for c in tree.children]
    def level4(self, tree):
        log.info('level4')
        log.info(dir(tree))
        [self.visit(c) for c in tree.children]
    def level3(self, tree):
        log.info('level3')
        log.info(type(tree))
        [self.visit(c) for c in tree.children]
    def level2(self, tree):
        log.info('level2')
        [self.visit(c) for c in tree.children]
    def level1(self, tree):
        log.info('level1')
        [self.visit(c) for c in tree.children]
    def atom(self, tree):
        log.info('atom')
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
