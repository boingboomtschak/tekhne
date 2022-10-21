#! /usr/bin/env python3
import argparse, os, logging, sys
import coloredlogs
from colorama import Fore, Back, Style
from lark import Lark, tree

# Setting up argument parser
argparser = argparse.ArgumentParser(description="CUDA to WGSL transpiler")
argparser.add_argument("input", help="Path to input .cu file")
argparser.add_argument("-o", "--output", help="Path to output .wgsl file")
argparser.add_argument("-d", "--debug", action="store_true", help="Show debug information")
argparser.add_argument("-f" ,"--file-log", action="store_true", help="Store logs to 'tekhne.log'")
argparser.add_argument("-t", "--parse-tree", action="store_true", help="Save parse tree to 'parse-tree.png'")
args = vars(argparser.parse_args())

# Setting up logger
log = logging.getLogger(__name__)
streamFmt=f'{Fore.CYAN}[{Fore.GREEN}tekhne{Fore.CYAN}]{Style.RESET_ALL} %(message)s'
logLevel = 'DEBUG' if args.get('debug') else 'INFO'
coloredlogs.install(fmt=streamFmt, level=logLevel, logger=log)
if args.get("file_log"):
    log.debug("Logging to 'tekhne.log'...")
    fileFmt=f'[tekhne] %(asctime)s : %(message)s'
    fileHandler = logging.FileHandler('tekhne.log')
    fileFormatter = logging.Formatter(fileFmt)
    fileHandler.setFormatter(fileFormatter)
    log.addHandler(fileHandler)

input_path = os.path.abspath(args['input'])
input_fname = os.path.basename(args['input'])
output_path = args['output'] if args['output'] else os.path.splitext(input_fname)[0] + '.wgsl'

log.debug("Reading input file...")
cuda = ''
try:
    with open(input_path, 'r') as f:
        cuda = f.read()
except FileNotFoundError:
    log.error(f'\'{input_fname}\' not found in \'{os.path.dirname(input_path)}\'!')
    sys.exit(1)    
except PermissionError:
    log.error(f'No permission to read \'{input_path}\'!')
    sys.exit(1)
except OSError:
    log.error(f'OS error reading \'{input_path}\'')
except Exception as e:
    log.error(f'Unknown error reading \'{input_path}\'!')
    sys.exit(1)

# Setting up CUDA grammar
grammar = '''
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
     | expression
?level1 : atom ("++"|"--")?
       | level1 "(" (atom ("," atom)*)? ")"
       | level1 "[" atom "]"
       | level1 "." atom
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

type : CNAME 
ptrtype : CNAME "*"

cudaspec : ("__shared__"|"__global__"|"__device__")

declaration : cudaspec? type CNAME ("[" expression "]")* ("=" expression)? ";"
            | cudaspec? type CNAME ("," CNAME)* ";"

conditional : "if" "(" expression ")" (("{" statement* "}")|statement) ("else" ("{" statement* "}")|statement)*

while_loop : "while" "(" expression ")" (("{" statement* "}")|statement)

for_loop : "for" "(" (declaration|assignment) expression ";" expression ")" (("{" statement* "}")|statement)

statement : for_loop
          | while_loop
          | conditional
          | declaration
          | expression ";"
          | assignment

argument : (type|ptrtype) CNAME 

kerneldecl : CNAME "(" argument ("," argument)* ")" 

kernelspec : "__global__" type kerneldecl "{" statement* "}"

start : kernelspec*
'''

# Setting up Lark parser
log.debug("Setting up parser...")
parser = Lark(grammar, ambiguity='explicit')

log.debug("Parsing input...")
parsed = parser.parse(cuda)

if args['parse_tree']:
       log.debug("Generating parse tree to 'parse-tree.png'")
       tree.pydot__tree_to_png(parsed, 'parse-tree.png')
       log.debug("Parse tree generated.")

log.debug("Exiting...")