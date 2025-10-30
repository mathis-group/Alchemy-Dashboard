import re
from dataclasses import dataclass, field
from typing import Union, List, Optional, Dict, Any
from enum import Enum
import json


#3 different node types
# either a lamba, a connector, or simple variable
class NodeType(Enum):
   LAMBDA = "lambda"
   APPLICATION = "application"
   VAR = "var"


# blueprint for each node type
@dataclass
class ASTNode:
   node_type: NodeType


# blueprint for a single variable (x)
@dataclass
class VariableNode(ASTNode):
   name: str
   node_type: NodeType = field(init=False, default=NodeType.VAR)
   # no child
   children: List['ASTNode'] = field(init=False, default_factory=list)


# blueprint for lambda
@dataclass
class LambdaNode(ASTNode):
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
 
   def __init__(self, expression: str):
        
       self.tokens = self.tokenize(expression)
       # point to the very first part of the expression
       self.pos = 0


#turns expression into list of tokens
   def tokenize(self, expression: str) -> List[str]:
       # seperating parenthesis with spaces
       expression = expression.replace('(', ' ( ').replace(')', ' ) ')
       #add each found token into list 
       tokens = re.findall(r'[λ\\]|[a-zA-Z0-9]+|\(|\)|\.', expression)
       return tokens




   #look at current token 
   def peek(self) -> Optional[str]:
       return self.tokens[self.pos] if self.pos < len(self.tokens) else None


   #move to next position 
   def _consume(self, expected: Optional[str] = None) -> str:
       if self.pos >= len(self.tokens):
           raise ValueError("Unexpected end")
       token = self.tokens[self.pos]
       if expected and token != expected:
           raise ValueError(f"Expected '{expected}' but found '{token}'")
       self.pos += 1
       return token


       # build expression into the blueprint object
   def parse(self) -> Optional[ASTNode]:
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
       #get first peice of sequence
       left_node = self.parse_chooser()
       #while there is another piece to the right, put them together
       while self.peek() and self.peek() not in [')']:
           right_node = self.parse_chooser()
           left_node = AppNode(left_node, right_node)
       #return the sequence
       return left_node


   #decide which rule to follow,
   def parse_chooser(self) -> ASTNode:
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
       self._consume()
       var_name = self._consume()
       if not re.match(r'^[a-zA-Z0-9]+$', var_name):
           raise ValueError(f"Invalid variable name: {var_name}")
       self._consume('.')
       body = self._parse_expression()
       return LambdaNode(var_name, body)


#creates vairable node object x
   def _parse_variable(self) -> VariableNode:
       name = self._consume()
       return VariableNode(name)










#for bokeh visualization
def getColors(node: Optional[ASTNode]) -> Dict[str, str]:
  
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




