
#ASTGen.py - This file serves as the blueprint for turning a lambda
# expression string into a tree of nodes (an abstract syntax tree, or AST)
"""
Parses lambda-calculus expressions into syntax trees for drawing.

Used by plotting.ASTvisualizer to draw an expression as a tree.

Accepted syntax:
    Variables     letters and digits, e.g. x, f, x1
    Lambda        λx.body  or  \\x.body   (the body extends as far right as
                  possible, so \\x.f x means \\x.(f x))
    Application   f x  (written side by side; groups from the left, so
                  a b c means (a b) c)
    Parentheses   for grouping, e.g. (\\x.x) y

Example: LambdaParser("(\\\\x.x) y").parse() builds this tree:

        App
       /   \\
     λx     y
     |
     x

Tree node types:
    VariableNode  a variable; no children
    LambdaNode    a function \\var.body; one child (the body)
    AppNode       an application "function arg"; two children

Run this file directly (python -m alchemy_dashboard.ASTGen) to parse a few
test expressions.
"""
import re
from dataclasses import dataclass, field
from typing import Union, List, Optional, Dict, Any
from enum import Enum
import json

#3 different node types
# either a lamba, a connector, or simple variable
class NodeType(Enum):
   """The kind of node in the tree."""
   LAMBDA = "lambda"
   APPLICATION = "application"
   VAR = "var"


# blueprint for each node type
@dataclass
class ASTNode:
   """Base class for all tree nodes.

   Every node has a `children` list (used to walk the tree) and a `name`
   (the text shown on the node when it is drawn).
   """
   node_type: NodeType


# blueprint for a single variable (x)
@dataclass
class VariableNode(ASTNode):
   """A variable such as x. Always a leaf (no children)."""
   name: str
   node_type: NodeType = field(init=False, default=NodeType.VAR)
   # no child
   children: List['ASTNode'] = field(init=False, default_factory=list)


# blueprint for lambda
@dataclass
class LambdaNode(ASTNode):
   """A function \\var.body. Its only child is the body.

   Its display name is "λ" + the variable, e.g. "λx".
   """
   #  bound var
   var: str
 
   body: 'ASTNode'
   node_type: NodeType = field(init=False, default=NodeType.LAMBDA)
   # one child
   children: List['ASTNode'] = field(init=False)


   # populate children
   def __post_init__(self):
       self.children = [self.body]


   @property
   def name(self):
       return f"λ{self.var}"


# blue print for application (f -> x)
@dataclass
class AppNode(ASTNode):
   """An application: `function` applied to `arg`, written "function arg".

   Children are [function, arg]. Its display name is "App".
   """
   # left
   function: 'ASTNode'
   # right
   arg: 'ASTNode'
   node_type: NodeType = field(init=False, default=NodeType.APPLICATION)
   # two children (left,right)
   children: List['ASTNode'] = field(init=False)
   name: str = "App"


   def __post_init__(self):
       self.children = [self.function, self.arg]


# the main parser, turns a string into an object
class LambdaParser:
   """Turns a lambda expression string into a tree of nodes.

   Usage:
       tree = LambdaParser("\\\\x.x y").parse()

   A new parser is needed for each expression. parse() raises ValueError
   if the expression is invalid.
   """

   def __init__(self, expression: str):
        
       self.tokens = self.tokenize(expression)
       # point to the very first part of the expression
       self.pos = 0


#turns expression into list of tokens
   def tokenize(self, expression: str) -> List[str]:
       """Split the expression into tokens: λ or \\, names, (, ), and ".".

       Example: "(\\\\x.x) y" -> ["(", "\\\\", "x", ".", "x", ")", "y"]

       Any other character (spaces, underscores, +, etc.) is silently
       skipped, so "x_1" becomes the two tokens "x" and "1".
       """
       # separating parentheses with spaces
       expression = expression.replace('(', ' ( ').replace(')', ' ) ')
       #add each found token into list 
       tokens = re.findall(r'[λ\\]|[a-zA-Z0-9]+|\(|\)|\.', expression)
       return tokens




   #look at current token 
   def peek(self) -> Optional[str]:
       """Return the current token without moving past it (None at the end)."""
       return self.tokens[self.pos] if self.pos < len(self.tokens) else None


   #move to next position 
   def _consume(self, expected: Optional[str] = None) -> str:
       """Return the current token and move to the next one.

       If `expected` is given and the token doesn't match, raises ValueError.
       """
       if self.pos >= len(self.tokens):
           raise ValueError("Unexpected end")
       token = self.tokens[self.pos]
       if expected and token != expected:
           raise ValueError(f"Expected '{expected}' but found '{token}'")
       self.pos += 1
       return token


       # build expression into the blueprint object
   def parse(self) -> Optional[ASTNode]:
       """Parse the whole expression and return the root node of the tree.

       Returns None for an empty expression. Raises ValueError if the
       expression is invalid.
       """
       # no tokens, do nothing
       if not self.tokens:
           return None
      
       #call parse expresison function
       ast = self._parse_expression()
       #check if there are any leftover pieces after building
       if self.peek() is not None:
           raise ValueError(f"Extra characters detected: {self._peek()}")
       return ast




   def _parse_expression(self) -> ASTNode:
       """Parse a sequence of terms and join them into applications.

       Terms are joined from the left: "a b c" becomes App(App(a, b), c).
       Stops at a ")" or at the end of the input.
       """
       #get first piece of sequence
       left_node = self.parse_chooser()
       #while there is another piece to the right, put them together
       while self.peek() and self.peek() not in [')']:
           right_node = self.parse_chooser()
           left_node = AppNode(left_node, right_node)
       #return the sequence
       return left_node


   #decide which rule to follow,
   def parse_chooser(self) -> ASTNode:
       """Parse one term, based on the current token.

       λ or \\ -> a lambda, "(" -> a group in parentheses, a name -> a
       variable. Anything else raises ValueError.
       """
       token = self.peek()
       #parse lambda
       if token in ['λ', '\\']:
           return self._parse_lambda()
       if token == '(':
           #parse parenthesized group
           self._consume('(')
           expression = self._parse_expression()
           self._consume(')')
           return expression
       #parse single variable name
       if token and re.match(r'^[a-zA-Z0-9]+$', token):
           return self._parse_variable()
       raise ValueError(f"Unexpected token: {token}")




#creates lambda node object
   def _parse_lambda(self) -> LambdaNode:
       """Parse "λvar.body" (or "\\var.body").

       The body is everything up to the closing ")" or the end, so
       "\\x.f x" is \\x.(f x), not (\\x.f) x.
       """
       self._consume()
       var_name = self._consume()
       if not re.match(r'^[a-zA-Z0-9]+$', var_name):
           raise ValueError(f"Invalid variable name: {var_name}")
       self._consume('.')
       body = self._parse_expression()
       return LambdaNode(var_name, body)


#creates variable node object x
   def _parse_variable(self) -> VariableNode:
       name = self._consume()
       return VariableNode(name)










#for bokeh visualization
def getColors(node: Optional[ASTNode]) -> Dict[str, str]:
   """Give each variable name in the tree its own color for drawing.

   Both lambda variables (the x in \\x.) and plain variables are included,
   so a lambda and the variables it binds get the same color. Names are
   sorted alphabetically before colors are assigned, so the result is the
   same every time. With more than 8 names, colors repeat.

   Note: two different variables with the same name (e.g. the two x's in
   (\\x.x) (\\x.x)) get the same color.

   Args:
       node: Root of the tree, or None.

   Returns:
       dict: Variable name -> color name, e.g. {"x": "blue", "y": "red"}.
   """
   variables = set()
  #traverse through AST and add unique objects to set
   def collect_var(n: ASTNode):
       if isinstance(n, VariableNode):
           variables.add(n.name)
       elif isinstance(n, LambdaNode):
           variables.add(n.var)
           if n.body: collect_var(n.body)
       elif isinstance(n, AppNode):
           #look at left and right func
           collect_var(n.function)
           collect_var(n.arg)
  
   if node:
       collect_var(node)
  
   colors = [
       "blue",
        "red",
        "green",
        "orange",
        "purple",
        "teal",
        "pink",
        "brown"
   ]
  
   color_map = {}
   sorted_var = sorted(list(variables)) 

   for i, var in enumerate(sorted_var):
       #color each unique var in sorted var
       color_map[var] = colors[i % len(colors)]
      
   return color_map


# Quick manual test: parses a few expressions and prints the resulting trees
if __name__ == "__main__":

   test_expressions = [
       "x",
       "\\y.y",
       "f x",
       "(λx.x) y",
       "λf.λx.f (f x)",
       "a b c d"
   ]


   print("Running Tests....")
   for i, expr_str in enumerate(test_expressions):
       print(f"\n{i+1}. Parsing Expression: '{expr_str}'")
       try:
         
           parser = LambdaParser(expr_str)
           ast = parser.parse()
           print(f"   Success! AST: {ast}")
       except ValueError as e:
           print(f"  Failed!: {e}")




