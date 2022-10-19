#! /usr/bin/python3
import argparse, os, logging, sys
import coloredlogs
from colorama import Fore, Back, Style
from lark import Lark

# Setting up argument parser
parser = argparse.ArgumentParser(description="CUDA to WGSL transpiler")
parser.add_argument("input", help="Path to input .cu file")
parser.add_argument("-o", "--output", help="Path to output .wgsl file")
parser.add_argument("-d", "--debug", action="store_true", help="Show debug information")
parser.add_argument("-f" ,"--file-log", action="store_true", help="Store logs to 'tekhne.log'")
args = vars(parser.parse_args())

# Setting up logger
log = logging.getLogger(__name__)
streamFmt=f'{Fore.CYAN}[{Fore.GREEN}tekhne{Fore.CYAN}]{Style.RESET_ALL} %(message)s'
logLevel = 'DEBUG' if args.get('debug') else 'INFO'
coloredlogs.install(fmt=streamFmt, level=logLevel, logger=log)
if args.get("file_log"):
    log.info("Logging to 'tekhne.log'...")
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
%ignore WS

%import common.CNAME
%import common.INT
%import common.SIGNED_INT
%import common.DECIMAL

BOOLEAN : "true"
        | "false"

atom : "(" expression ")"
     | INT
     | SIGNED_INT
     | DECIMAL
     | CNAME
     | BOOLEAN

level1 : atom "++"
       | atom "--"
       | atom "(" (expression ("," expression)*)? ")"
       | atom "[" expression "]"
       | atom "." CNAME
       | atom

level2 : "++" level1
       | "+" level1
       | "-" level1
       | "--" level1
       | "!" level1
       | "*" level1
       | "~" level1
       | level1

level3 : level3 "*" level2
       | level3 "/" level2
       | level3 "%" level2
       | level2

level4 : level4 "+" level3
       | level4 "-" level3
       | level3

level5 : level5 "<<" level4
       | level5 ">>" level4
       | level4

level6 : level6 "<" level5
       | level6 ">" level5
       | level6 "<=" level5
       | level6 ">=" level5
       | level5

level7 : level7 "==" level6
       | level7 "!=" level6
       | level6

level8 : level8 "&" level7
       | level7

level9 : level9 "^" level8
       | level8

level10 : level10 "|" level9
        | level9

level11 : level11 "&&" level10
        | level10

level12 : level12 "||" level11
        | level11

expression : level12

lvalue : CNAME "[" expression "]"
       | CNAME "." CNAME
       | CNAME

assignment  : lvalue "=" expression ";"
            | lvalue "+=" expression ";"
            | lvalue "-=" expression ";"
            | lvalue "*=" expression ";"
            | lvalue "/=" expression ";"

type : CNAME "*"? 

declaration : type CNAME ";"
            | type CNAME "=" expression ";"

conditional : "if" "(" expression ")" "{" statement* "}"
            | "if" "(" expression ")" statement

while_loop : "while" "(" expression ")" "{" statement* "}"

for_loop : "for" "(" declaration expression expression ")" "{" statement* "}"
         | "for" "(" declaration expression expression ")" statement

statement : for_loop
          | while_loop
          | conditional
          | declaration
          | expression
          | assignment

argument: type CNAME 

kernelDecl: CNAME "(" argument ("," argument)* ")" 

kernelSpec: "__global__" type kernelDeclaration "{" statement* "}"

start: kernelSpec

'''

# Setting up Lark parser
log.debug("Setting up parser...")
parser = Lark(grammar)

tree = parser.parse(cuda)
print(tree.pretty())