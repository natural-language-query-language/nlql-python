from typing import List, Iterator, Dict, Any, Optional
from collections import defaultdict
import time

from ..parser.ast import (
    Query, Condition, BinaryCondition,
    UnaryCondition, OperatorCondition, ComparisonCondition
)
from ..lexer.token import TokenType
from ..utils.text_unit import TextUnit, TextUnitizer
from ..exceptions import ExecutionError
from .operators import OperatorFactory
from ..optimization.cache import QueryCache, QueryOptimizer, QueryStatistics
from ..optimization.index import TextIndex

class QueryExecutor:
    """
    Executes queries on text data.
    
    This class is responsible for executing parsed query ASTs against text data,
    applying conditions, grouping, ordering, and limiting results.
    """
    
    def __init__(
        self,
        operator_factory: Optional[OperatorFactory] = None,
        use_cache: bool = True,
        use_index: bool = True
    ):
        """
        Initialize the query executor.
        
        Args:
            operator_factory (OperatorFactory, optional): Factory for creating operators
            use_cache (bool): Whether to use query caching
            use_index (bool): Whether to use text indexing
        """
        self.operator_factory = operator_factory or OperatorFactory()
        self.text_unitizer = TextUnitizer()
        
        # Optimization components
        self.use_cache = use_cache
        self.use_index = use_index
        self.cache = QueryCache() if use_cache else None
        self.index = TextIndex() if use_index else None
        self.optimizer = QueryOptimizer()
        self.stats = QueryStatistics()

    def execute(
        self,
        query: Query,
        text: str,
        metadata_extractors: Dict[str, callable] = None
    ) -> Iterator[TextUnit]:
        """
        Execute a query against text data.
        
        Args:
            query (Query): The query AST to execute
            text (str): The text to query against
            metadata_extractors (Dict[str, callable], optional): Functions to extract metadata
            
        Returns:
            Iterator[TextUnit]: Matching text units
            
        Raises:
            ExecutionError: If there's an error during query execution
        """
        try:
            # Check cache first
            if self.use_cache:
                cached_results = self.cache.get(str(query), text)
                if cached_results is not None:
                    self.stats.record_cache_hit()
                    return iter(cached_results)
                self.stats.record_cache_miss()
            
            # Start execution timer
            start_time = time.time()
            
            # Optimize query
            optimized_query = self.optimizer.optimize_query(query)
            
            # Split text into units
            units = list(self.text_unitizer.split_into_units(
                text,
                optimized_query.unit,
                metadata_extractors
            ))
            
            # Index units if needed
            if self.use_index and self.index:
                self.index.clear()  # Clear previous index
                for unit in units:
                    self.index.add_unit(unit)
            
            # Execute query
            results = self._execute_optimized(
                optimized_query,
                units
            )
            
            # Convert to list for caching
            results_list = list(results)
            
            # Cache results
            if self.use_cache:
                self.cache.set(str(query), text, results_list)
            
            # Record execution time
            execution_time = time.time() - start_time
            self.stats.record_query_time(str(query), execution_time)
            
            return iter(results_list)
            
        except Exception as e:
            raise ExecutionError(f"Error executing query: {str(e)}")

    def _execute_optimized(
        self,
        query: Query,
        units: List[TextUnit]
    ) -> Iterator[TextUnit]:
        """
        Execute optimized query.
        
        Args:
            query (Query): The optimized query to execute
            units (List[TextUnit]): The text units to query against
            
        Returns:
            Iterator[TextUnit]: Matching text units
        """
        # Apply conditions if any
        if query.conditions:
            units = self._filter_units(units, query.conditions)
        
        # Apply grouping if specified
        if query.group_by:
            units = self._group_units(units, query.group_by)
        
        # Apply ordering if specified
        if query.order_by:
            units = self._order_units(units, query.order_by)
        
        # Apply limit if specified
        if query.limit is not None:
            units = self._limit_units(units, query.limit)
        
        yield from units

    def _filter_units(
        self,
        units: Iterator[TextUnit],
        condition: Condition
    ) -> Iterator[TextUnit]:
        """
        Filter text units based on a condition.
        
        Args:
            units (Iterator[TextUnit]): Text units to filter
            condition (Condition): Condition to apply
            
        Returns:
            Iterator[TextUnit]: Filtered text units
        """
        for unit in units:
            if self._evaluate_condition(unit, condition):
                yield unit

    def _evaluate_condition(self, unit: TextUnit, condition: Condition) -> bool:
        """
        Evaluate a condition for a text unit.
        
        Args:
            unit (TextUnit): Text unit to evaluate
            condition (Condition): Condition to evaluate
            
        Returns:
            bool: Whether the condition is met
            
        Raises:
            ExecutionError: If condition type is not supported
        """
        if isinstance(condition, BinaryCondition):
            if condition.operator == TokenType.AND:
                return (self._evaluate_condition(unit, condition.left) and
                       self._evaluate_condition(unit, condition.right))
            elif condition.operator == TokenType.OR:
                return (self._evaluate_condition(unit, condition.left) or
                       self._evaluate_condition(unit, condition.right))
            else:
                raise ExecutionError(f"Unsupported binary operator: {condition.operator}")

        elif isinstance(condition, UnaryCondition):
            if condition.operator == TokenType.NOT:
                return not self._evaluate_condition(unit, condition.operand)
            else:
                raise ExecutionError(f"Unsupported unary operator: {condition.operator}")

        elif isinstance(condition, OperatorCondition):
            operator = self.operator_factory.create_operator(condition.operator)
            return operator(unit, *condition.arguments)

        elif isinstance(condition, ComparisonCondition):
            operator = self.operator_factory.create_operator(condition.operator)
            return operator(unit, condition.right)

        else:
            raise ExecutionError(f"Unsupported condition type: {type(condition)}")

    def _group_units(
        self,
        units: Iterator[TextUnit],
        group_by: str
    ) -> Iterator[TextUnit]:
        """
        Group text units by a metadata field.
        
        Args:
            units (Iterator[TextUnit]): Text units to group
            group_by (str): Metadata field to group by
            
        Returns:
            Iterator[TextUnit]: Grouped text units
        """
        # Convert iterator to list since we need multiple passes
        units_list = list(units)
        if not units_list:
            return iter([])
            
        # Group units by the specified field
        groups = defaultdict(list)
        for unit in units_list:
            key = unit.metadata.get(group_by)
            if key is None:
                raise ExecutionError(f"Missing metadata field: {group_by}")
            groups[key].append(unit)
            
        # Yield units group by group
        for group_key in sorted(groups.keys()):
            yield from groups[group_key]

    def _order_units(
        self,
        units: Iterator[TextUnit],
        order_by: str
    ) -> Iterator[TextUnit]:
        """
        Order text units by a metadata field.
        
        Args:
            units (Iterator[TextUnit]): Text units to order
            order_by (str): Metadata field to order by
            
        Returns:
            Iterator[TextUnit]: Ordered text units
        """
        # Convert iterator to list for sorting
        units_list = list(units)
        if not units_list:
            return iter([])
            
        # Define sort key function
        def sort_key(unit: TextUnit) -> Any:
            value = unit.metadata.get(order_by)
            if value is None:
                raise ExecutionError(f"Missing metadata field: {order_by}")
            return value
            
        # Sort and return iterator
        return iter(sorted(units_list, key=sort_key))

    def _limit_units(
        self,
        units: Iterator[TextUnit],
        limit: int
    ) -> Iterator[TextUnit]:
        """
        Limit the number of text units.
        
        Args:
            units (Iterator[TextUnit]): Text units to limit
            limit (int): Maximum number of units to return
            
        Returns:
            Iterator[TextUnit]: Limited text units
        """
        for i, unit in enumerate(units):
            if i >= limit:
                break
            yield unit
