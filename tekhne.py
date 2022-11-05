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
?level1 : atom "++" -> inc
        | atom "--" -> dec
        | level1 "(" (expression ("," expression)*)? ")" -> call
        | level1 "[" expression "]" -> idx_access
        | level1 "." CNAME -> prop_access
        | atom
?level2 : "++" level1 -> pre_inc
        | "+" level1 -> pos
        | "--" level1 -> pre_dec
        | "-" level1 -> neg
        | "!" level1 -> log_not
        | "~" level1 -> bit_not
        | "*" level1 -> deref
        | level1
?level3 : level3 "*" level2 -> mult
        | level3 "/" level2 -> div
        | level3 "%" level2 -> mod
        | level2
?level4 : level4 "+" level3 -> plus 
        | level4 "-" level3 -> minus
        | level3 
?level5 : level5 "<<" level4 -> lshift
        | level5 ">>" level4 -> rshift
        | level4
?level6 : level6 "<" level5 -> lt
        | level6 ">" level5 -> gt
        | level6 "<=" level5 -> lte
        | level6 ">=" level5 -> gte
        | level5
?level7 : level7 "==" level6 -> eq
        | level7 "!=" level6 -> neq
        | level6
?level8 : level8 "&" level7 -> bit_and
        | level7
?level9 : level9 "^" level8 -> bit_xor
        | level8 
?level10 : level10 "|" level9 -> bit_or
         | level9
?level11 : level11 "&&" level10 -> log_and
         | level10
?level12 : level12 "||" level11 -> log_or
         | level11 
?expression : level12

?lvalue : CNAME (("[" expression "]")|("." CNAME))*

assignment : lvalue "=" expression ";"
           | lvalue "+=" expression ";" -> inc_assignment
           | lvalue "-=" expression ";" -> dec_assignment
           | lvalue "*=" expression ";" -> mul_assignment
           | lvalue "/=" expression ";" -> div_assignment

ctype : CNAME 
ptrtype : CNAME "*"

SHARED : "__shared__"
GLOBAL : "__global__"
DEVICE : "__device__"

expr_statement : expression ";"

declaration : SHARED? ctype CNAME ("[" expression "]")* ("=" expression)? ";"
            | SHARED? ctype CNAME ("," CNAME)+ ";" -> mult_declaration

conditional_else : "else" (("{" statement* "}")|statement)
conditional_if : "if" "(" expression ")" (("{" statement* "}")|statement)
conditional : conditional_if conditional_else*

while_loop : "while" "(" expression ")" (("{" statement* "}")|statement)

for_loop : "for" "(" (declaration|assignment) expression ";" expression ")" (("{" statement* "}")|statement)

statement : for_loop
           | while_loop
           | conditional
           | declaration
           | expr_statement
           | assignment

argument : (ctype|ptrtype) CNAME 

kerneldecl : CNAME "(" argument ("," argument)* ")" 

kernelspec : GLOBAL ctype kerneldecl "{" statement* "}"

start : kernelspec*
'''


'''
CodeGen todos:
- Scope tabbing
- Fix declaration newlining - still some weird issues with extra newlines around conditionals (in test.cu)
- Quick pre-visitor for builtins, need to track builtins inside kerneldecl and dump in fn main
  - Need to track builtins per kernel (may have multiple per input file) so no global tracking
    in code generator
- Uniform binding dumps for CUDA kernel args, also needs to happen in kerneldecl 
- Token-specific visitor rule to replace builtins (global translation dict like TYPE_REDEFS needed)
'''
class WGSLCodeGenerator:      
    TYPE_REDEFS = {
        'int' : 'i32',
        'float' : 'f32'
    }  
    def __init__(self):
        self.depth = 0
    def visit(self, tree):
        if isinstance(tree, Token):
            return str(tree)
        try:
            return getattr(self, tree.data)(tree)
        except AttributeError:
            return self.__default__(tree)
    def visit_children(self, children, joiner=""):
        return joiner.join([self.visit(c) for c in children])
    def __default__(self, tree):
        log.warning(f"Default visitor visiting {tree.data}!")
        return "".join([self.visit(c) for c in tree.children])
    def start(self, tree):
        return "".join([self.visit(c) for c in tree.children])
    def kernelspec(self, tree):
        buf = "@compute\n"
        buf += self.visit(tree.children[2])
        buf += "{\n" + self.visit_children(tree.children[3:]) + "}"
        return buf
    def kerneldecl(self, tree):
        return "fn main()" # See todos for changes here
    def for_loop(self, tree):
        buf = "for ("
        buf += self.visit(tree.children[0]) + " "
        buf += self.visit(tree.children[1]) + "; "
        buf += self.visit(tree.children[2]) + ")"
        if len(tree.children) > 4:
            buf += " {\n" + self.visit_children(tree.children[3:]) + "}"
        else:
            buf += "\n" + self.visit(tree.children[3]) 
        return buf
    def while_loop(self, tree):
        buf = "while (" + self.visit(tree.children[0]) + ") "
        if len(tree.children) > 2:
            buf += "{\n" + self.visit_children(tree.children[1:]) + "}"
        else:
            buf += "\n" + self.visit(tree.children[1])
        return buf
    def conditional_if(self, tree):
        buf = f'if ({self.visit(tree.children[0])})'
        if len(tree.children) > 2:
            buf += " {\n" + self.visit_children(tree.children[1:]) + "}"
        else:
            buf += "\n" + self.visit(tree.children[1])
        return buf
    def conditional_else(self, tree):
        buf = 'else'
        if len(tree.children) > 1:
            buf += " {\n" + self.visit_children(tree.children) + "}"
        else:
            buf += "\n" + self.visit(tree.children[0])
        return buf
    def conditional(self, tree):
        return self.visit_children(tree.children)
    def declaration(self, tree):
        buf = f"var {self.visit(tree.children[1])} : {self.visit(tree.children[0])}"
        if len(tree.children) > 2:
            buf += " = " + self.visit(tree.children[2])
        buf += ";"
        return buf
    def statement(self, tree):
        return self.visit(tree.children[0]) + "\n"
    def ctype(self, tree):
        tok = self.visit(tree.children[0])
        if tok in self.TYPE_REDEFS:
            return self.TYPE_REDEFS[tok]
        return tok
    def assignment(self, tree):
        return f'{self.visit(tree.children[0])} = {self.visit(tree.children[1])};'
    def inc_assignment(self, tree):
        return f'{self.visit(tree.children[0])} += {self.visit(tree.children[1])};'
    def dec_assignment(self, tree):
        return f'{self.visit(tree.children[0])} -= {self.visit(tree.children[1])};'
    def mul_assignment(self, tree):
        return f'{self.visit(tree.children[0])} *= {self.visit(tree.children[1])};'
    def div_assignment(self, tree):
        return f'{self.visit(tree.children[0])} /= {self.visit(tree.children[1])};'
    def expr_statement(self, tree):
        return self.visit(tree.children[0]) + ";"
    def inc(self, tree):
        return self.visit(tree.children[0]) + "++"
    def dec(self, tree):
        return self.visit(tree.children[0]) + "--"
    def call(self, tree):
        buf = self.visit(tree.children[0]) 
        buf += "(" + self.visit_children(tree.children[1:], joiner=", ") + ")"
        return buf
    def idx_access(self, tree):
        return self.visit(tree.children[0]) + "[" + self.visit(tree.children[1]) + "]"
    def prop_access(self, tree):
        return self.visit(tree.children[0]) + "." + self.visit(tree.children[1])
    def pre_inc(self, tree):
        return "++" + self.visit(tree.children[0])
    def pos(self, tree):
        return "+" + self.visit(tree.children[0])
    def pre_dec(self, tree):
        return "--" + self.visit(tree.children[0])
    def neg(self, tree):
        return "-" + self.visit(tree.children[0])
    def log_not(self, tree):
        return "!" + self.visit(tree.children[0])
    def bit_not(self, tree):
        return "~" + self.visit(tree.children[0])
    def deref(self, tree):
        return "*" + self.visit(tree.children[0])
    def mult(self, tree):
        return self.visit(tree.children[0]) + " * " + self.visit(tree.children[1])
    def div(self, tree):
        return self.visit(tree.children[0]) + " / " + self.visit(tree.children[1])
    def mod(self, tree):
        return self.visit(tree.children[0]) + " % " + self.visit(tree.children[1])
    def plus(self, tree):
        return self.visit(tree.children[0]) + " + " + self.visit(tree.children[1])
    def minus(self, tree):
        return self.visit(tree.children[0]) + " - " + self.visit(tree.children[1])
    def lshift(self, tree):
        return self.visit(tree.children[0]) + " << " + self.visit(tree.children[1])
    def rshift(self, tree):
        return self.visit(tree.children[0]) + " >> " + self.visit(tree.children[1])
    def lt(self, tree):
        return self.visit(tree.children[0]) + " < " + self.visit(tree.children[1])
    def gt(self, tree):
        return self.visit(tree.children[0]) + " > " + self.visit(tree.children[1])
    def lte(self, tree):
        return self.visit(tree.children[0]) + " <= " + self.visit(tree.children[1])
    def gte(self, tree):
        return self.visit(tree.children[0]) + " >= " + self.visit(tree.children[1])
    def eq(self, tree):
        return self.visit(tree.children[0]) + " == " + self.visit(tree.children[1])
    def neq(self, tree):
        return self.visit(tree.children[0]) + " != " + self.visit(tree.children[1])
    def bit_and(self, tree):
        return self.visit(tree.children[0]) + " & " + self.visit(tree.children[1])
    def bit_xor(self, tree):
        return self.visit(tree.children[0]) + " ^ " + self.visit(tree.children[1])
    def bit_or(self, tree):
        return self.visit(tree.children[0]) + " | " + self.visit(tree.children[1])
    def log_and(self, tree):
        return self.visit(tree.children[0]) + " && " + self.visit(tree.children[1])
    def log_or(self, tree):
        return self.visit(tree.children[0]) + " || " + self.visit(tree.children[1])

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
log.debug("Generated code:")
if args.get('debug'):
    sys.stdout.write(WGSLCodeGenerator().visit(parsed) + "\n")

log.debug("Exiting...")
