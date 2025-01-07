from dataclasses import dataclass
from typing import List, Optional, Any
from ..lexer.token import TokenType

@dataclass
class ASTNode:
    """Base class for all AST nodes."""
    pass

@dataclass
class Query(ASTNode):
    """Represents a complete query."""
    unit: TokenType
    source: Optional[str]
    conditions: Optional['Condition']
    group_by: Optional[str]
    order_by: Optional[str]
    limit: Optional[int]

@dataclass
class Condition(ASTNode):
    """Base class for all conditions."""
    pass

@dataclass
class BinaryCondition(Condition):
    """Represents a binary condition (e.g., AND, OR)."""
    operator: TokenType
    left: Condition
    right: Condition

@dataclass
class UnaryCondition(Condition):
    """Represents a unary condition (e.g., NOT)."""
    operator: TokenType
    operand: Condition

@dataclass
class OperatorCondition(Condition):
    """Represents an operator condition (e.g., CONTAINS, STARTS_WITH)."""
    operator: TokenType
    arguments: List[Any]

@dataclass
class ComparisonCondition(Condition):
    """Represents a comparison condition (e.g., LENGTH > 10)."""
    operator: TokenType
    left: Any
    right: Any

class ASTVisitor:
    """Base visitor class for traversing the AST."""
    
    def visit(self, node: ASTNode):
        """Visit a node."""
        method_name = f'visit_{node.__class__.__name__}'
        method = getattr(self, method_name, self.generic_visit)
        return method(node)
    
    def generic_visit(self, node: ASTNode):
        """Called if no explicit visitor function exists for a node."""
        raise NotImplementedError(
            f'No visit_{node.__class__.__name__} method exists')