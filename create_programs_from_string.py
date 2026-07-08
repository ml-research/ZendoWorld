import re
from program import Function

# Define function to parse the DSL query
def parse_dsl_string_new(rule):
    """
    Parse a DSL-like rule, extracting the predicate and arguments.

    Supports:
      AT_LEAST_1 1 IS_RED
      AT_LEAST_1(1, IS_RED)
      (AT_LEAST_1 1 IS_RED)
      (AND (EVEN_1 IS_RED) (ODD_2 IS_BLOCK IS_PYRAMID))
      EITHER_OR 1 2
    """
    rule = rule.strip()

    # Strip outer parentheses if they wrap the whole expression
    if rule.startswith("(") and rule.endswith(")"):
        depth = 0
        for i, ch in enumerate(rule):
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
            if depth == 0 and i != len(rule) - 1:
                break
        else:
            rule = rule[1:-1].strip()

    # Predicate is first token (until space or '(')
    predicate = ""
    rest = ""

    for i, ch in enumerate(rule):
        if ch.isspace() or ch == "(":
            predicate = rule[:i]
            rest = rule[i:]
            break
    else:
        return rule, []

    predicate = predicate.strip()
    rest = rest.strip()

    # Remove surrounding parentheses from arguments
    if rest.startswith("(") and rest.endswith(")"):
        rest = rest[1:-1].strip()

    args = []
    cur = ""
    depth = 0

    for ch in rest:
        if ch in "([": depth += 1
        elif ch in ")]": depth -= 1

        if (ch == "," or ch.isspace()) and depth == 0:
            if cur.strip():
                args.append(cur.strip())
                cur = ""
        else:
            cur += ch

    if cur.strip():
        args.append(cur.strip())

    return predicate, args
# Define function to parse the DSL query
def parse_dsl_string(rule):
    """
    Parse a DSL-like rule, extracting the predicate and arguments.
    Example input: "at_least(yellow, pyramid, 1, Structure)"
    """
    rule = rule.strip()
    
    # --- Case 1: Lisp-style, e.g. (AND (AT_LEAST_1 1 IS_PYRAMID) ...)
    if rule.startswith("(") and rule.endswith(")"):
        # Strip outer parentheses
        inner = rule[1:-1].strip()
    else:
        inner = rule
    # First token is predicate
    tokens = inner.split(maxsplit=1)
    if len(tokens) == 1:
        return tokens[0], []
    predicate, rest = tokens
    # Split rest into top-level arguments
    args = []
    paren_depth = 0
    cur = ""
    for ch in rest:
        if ch == "(":
            paren_depth += 1
            cur += ch
        elif ch == ")":
            paren_depth -= 1
            cur += ch
        elif ch.isspace() and paren_depth == 0:
            if cur.strip():
                args.append(cur.strip())
                cur = ""
        else:
            cur += ch
    if cur.strip():
        args.append(cur.strip())
    return predicate, args


# Example function to convert the Prolog query into DSL semantics
def convert_string_to_dsl(rule_string, cfg) -> Function:
    predicate, args = parse_dsl_string(rule_string)
    
    if predicate == 'ALL_THREE_SHAPES':
        return Function(cfg.lookup['ALL_THREE_SHAPES'], [])
    if predicate == 'ALL_THREE_COLORS':
        return Function(cfg.lookup['ALL_THREE_COLORS'], [])
    if predicate == 'AT_LEAST_2':
        if len(args) == 3:  # AT_LEAST with color, shape, and count
            count, pred1, pred2 = args
            count = int(count)  # Convert count to integer
            # Create Function with valid Program arguments
            return Function(cfg.lookup['AT_LEAST_2'], [
                cfg.lookup[f'constant_{count}'],
                Function(cfg.lookup[pred1], []), 
                Function(cfg.lookup[pred2], []),
            ])
        else:
            raise ValueError(f"Invalid number of arguments for at_least: {len(args)}")
    if predicate == 'AT_LEAST_1':
        if len(args) == 2:  # at_least with color and count
            count, pred1 = args
            count = int(count)  # Convert count to integer
            return Function(cfg.lookup['AT_LEAST_1'], [
                cfg.lookup[f'constant_{count}'],
                Function(cfg.lookup[pred1], []),
            ])
        else:
            raise ValueError(f"Invalid number of arguments for at_least: {len(args)}")
        
    if predicate == 'MAJORITY_2':
        if len(args) == 2:  # AT_LEAST with color, shape, and count
            pred1, pred2 = args
            # Create Function with valid Program arguments
            return Function(cfg.lookup['MAJORITY_2'], [
                Function(cfg.lookup[pred1], []), 
                Function(cfg.lookup[pred2], []),
            ])
        else:
            raise ValueError(f"Invalid number of arguments for at_least: {len(args)}")
    if predicate == 'MAJORITY_1':
        if len(args) == 1:  # at_least with color and count
            pred1 = args[0]
            return Function(cfg.lookup['MAJORITY_1'], [
                Function(cfg.lookup[pred1], []),
            ])
        else:
            raise ValueError(f"Invalid number of arguments for at_least: {len(args)}")
    elif predicate == 'ALL_2':
        if len(args) == 2:  # AT_LEAST with color, shape, and count
            pred1, pred2 = args
            # Create Function with valid Program arguments
            return Function(cfg.lookup['ALL_2'], [
                Function(cfg.lookup[pred1], []), 
                Function(cfg.lookup[pred2], []),
            ])
        else:
            raise ValueError(f"Invalid number of arguments for at_least: {len(args)}")
    elif predicate == 'ALL_1':
        if len(args) == 1:  # at_least with color and count
            pred1 = args[0]
            return Function(cfg.lookup['ALL_1'], [
                Function(cfg.lookup[pred1], []),
            ])
        else:
            raise ValueError(f"Invalid number of arguments for at_least: {len(args)}")
            
    elif predicate == 'SAME_AMOUNT':
        if len(args) == 2:
            pred1, pred2 = args
            # Create Function with valid Program arguments
            return Function(cfg.lookup['SAME_AMOUNT'], [
                Function(cfg.lookup[pred1], []), 
                Function(cfg.lookup[pred2], []),
            ])
        else:
            raise ValueError(f"Invalid number of arguments for same_amount: {len(args)}")
    elif predicate == 'EXACTLY_2':
        if len(args) == 3:  # exactly with color, shape, and count
            count, pred1, pred2 = args
            count = int(count)  # Convert count to integer
            # Create Function with valid Program arguments
            return Function(cfg.lookup['EXACTLY_2'], [
                cfg.lookup[f'constant_{count}'],
                Function(cfg.lookup[pred1], []), 
                Function(cfg.lookup[pred2], []),
            ])
        else:
            raise ValueError(f"Invalid number of arguments for exactly: {len(args)}")
    elif predicate == 'EXACTLY_1':
        if len(args) == 2:  # exactly with color and count
            count, pred1 = args
            count = int(count)  # Convert count to integer
            return Function(cfg.lookup['EXACTLY_1'], [
                cfg.lookup[f'constant_{count}'],
                Function(cfg.lookup[pred1], []), 
            ])
        else:
            raise ValueError(f"Invalid number of arguments for exactly: {len(args)}")

    elif predicate == 'ZERO_1':
        if len(args) == 1:
            pred = args[0]
            # Create Function with valid Program arguments
            return Function(cfg.lookup['ZERO_1'], [
                Function(cfg.lookup[pred], []),
            ])
        else:
            raise ValueError(f"Invalid number of arguments for zero: {len(args)}")
    elif predicate == 'ZERO_2':
        if len(args) == 2:
            pred1, pred2 = args
            return Function(cfg.lookup['ZERO_2'], [
                Function(cfg.lookup[pred1], []), 
                Function(cfg.lookup[pred2], []), 
            ])
        else:
            raise ValueError(f"Invalid number of arguments for zero: {len(args)}")

    elif predicate == 'EXCLUSIVELY':
        if len(args) == 1:  # exactly with color, shape, and count
            pred = args[0]
            # Create Function with valid Program arguments
            return Function(cfg.lookup['EXCLUSIVELY'], [ 
                Function(cfg.lookup[pred], []),
            ])
        else:
            raise ValueError(f"Invalid number of arguments for exclusively: {len(args)}")

    elif predicate == 'AND':
        # Handle 'and' predicate by combining two rules
        subquery0 = args[0].strip('[]')
        subquery1 = args[1].strip('[]')
        rule1 = convert_string_to_dsl(subquery0, cfg)
        rule2 = convert_string_to_dsl(subquery1, cfg)
        return Function(cfg.lookup['AND'], [rule1, rule2])
    
    elif predicate == 'OR':
        # Handle 'or' predicate by combining two rules
        subquery0 = args[0].strip('[]')
        subquery1 = args[1].strip('[]')
        rule1 = convert_string_to_dsl(subquery0, cfg)
        rule2 = convert_string_to_dsl(subquery1, cfg)
        return Function(cfg.lookup['OR'], [rule1, rule2])
    
    elif predicate == 'EITHER_OR':
        # Handle 'either_or' predicate
        n1, n2 = args
        n1 = int(n1)
        n2 = int(n2)
        return Function(cfg.lookup['EITHER_OR'], [cfg.lookup[f'constant_{n1}'], cfg.lookup[f'constant_{n2}']])
    
    elif predicate == 'ODD':
        # Handle 'odd_number_of' predicate
        if len(args) == 0:  # odd_number_of with no predicate
            return Function(cfg.lookup['ODD'], [])
        else:
            raise ValueError(f"Invalid number of arguments for odd_number_of: {len(args)}")
    elif predicate == 'ODD_1':
        if len(args) == 1:  # odd_number_of with one predicate
            attr = args[0]
            return Function(cfg.lookup['ODD_1'], [Function(cfg.lookup[attr], [])])
        else:
            raise ValueError(f"Invalid number of arguments for odd_number_of: {len(args)}")
    elif predicate == 'ODD_2':
        if len(args) == 2:  # odd_number_of with two predicates (e.g. touching)
            pred1, pred2 = args
            return Function(cfg.lookup['ODD_2'], [
                    Function(cfg.lookup[pred1], []), 
                    Function(cfg.lookup[pred2], []),
                ])
        else:
            raise ValueError(f"Invalid number of arguments for odd_number_of: {len(args)}")
    
    elif predicate == 'EVEN':
        if len(args) == 0:  # even_number_of with no predicate
            return Function(cfg.lookup['EVEN'], [])
        else:
            raise ValueError(f"Invalid number of arguments for even_number_of: {len(args)}")
    elif predicate == 'EVEN_1':
        if len(args) == 1:  # even_number_of with one predicate
            attr = args[0]
            return Function(cfg.lookup['EVEN_1'], [Function(cfg.lookup[attr], [])])
        else:
            raise ValueError(f"Invalid number of arguments for even_number_of: {len(args)}")
    elif predicate == 'EVEN_2':
        if len(args) == 2:  # even_number_of with two predicates (e.g. touching)
            pred1, pred2 = args
            return Function(cfg.lookup['EVEN_2'], [
                    Function(cfg.lookup[pred1], []), 
                    Function(cfg.lookup[pred2], []),
                ])
        else:
            raise ValueError(f"Invalid number of arguments for even_number_of: {len(args)}")
    
    elif predicate == 'AT_LEAST_INTERACTION':
        if len(args) == 2:
            count, interaction_args = args
            interaction, args = parse_dsl_string(interaction_args)
            pred1, pred2 = args
            count = int(count)  # Convert count to integer
            return Function(cfg.lookup['AT_LEAST_INTERACTION'], [
                cfg.lookup[f'constant_{count}'],
                Function(cfg.lookup[interaction.upper()],
                [
                    Function(cfg.lookup[pred1.upper()], []), 
                    Function(cfg.lookup[pred2.upper()], []),
                ])
            ])
        else:
            raise ValueError(f"Invalid number of arguments for at_least_interaction: {len(args)}, {args}")
    
    elif predicate == 'EXACTLY_INTERACTION':
        if len(args) == 2:
            count, interaction_args = args
            interaction, args = parse_dsl_string(interaction_args)
            pred1, pred2 = args
            count = int(count)  # Convert count to integer
            return Function(cfg.lookup['EXACTLY_INTERACTION'], [
                cfg.lookup[f'constant_{count}'],
                Function(cfg.lookup[interaction.upper()],
                [
                    Function(cfg.lookup[pred1.upper()], []), 
                    Function(cfg.lookup[pred2.upper()], []),
                ])
            ])
        else:
            raise ValueError(f"Invalid number of arguments for exactly_interaction: {len(args)}")
        
    elif predicate == 'ODD_INTERACTION':
        if len(args) == 1:
            interaction, args = parse_dsl_string(args[0])
            pred1, pred2 = args
            return Function(cfg.lookup['ODD_INTERACTION'], [
                Function(cfg.lookup[interaction.upper()],
                [
                    Function(cfg.lookup[pred1.upper()], []), 
                    Function(cfg.lookup[pred2.upper()], []),
                ])
            ])
        else:
            raise ValueError(f"Invalid number of arguments for odd_number_of_interaction: {len(args)}")
        
    elif predicate == 'EVEN_INTERACTION':
        # Handle 'even_number_of_interaction' predicate
        if len(args) == 1:
            interaction, args = parse_dsl_string(args[0])
            pred1, pred2 = args
            return Function(cfg.lookup['EVEN_INTERACTION'], [
                Function(cfg.lookup[interaction.upper()],
                [
                    Function(cfg.lookup[pred1.upper()], []), 
                    Function(cfg.lookup[pred2.upper()], []),
                ])
            ])
        else:
            raise ValueError(f"Invalid number of arguments for even_number_of_interaction: {len(args)}")
        
    elif predicate == 'MAJORITY_INTERACTION':
        # Handle 'even_number_of_interaction' predicate
        if len(args) == 1:
            interaction, args = parse_dsl_string(args[0])
            pred1, pred2 = args
            return Function(cfg.lookup['MAJORITY_INTERACTION'], [
                Function(cfg.lookup[interaction.upper()],
                [
                    Function(cfg.lookup[pred1.upper()], []), 
                    Function(cfg.lookup[pred2.upper()], []),
                ])
            ])
        else:
            raise ValueError(f"Invalid number of arguments for majority_interaction: {len(args)}")
    
    elif predicate == 'MORE_THAN':
        # Handle 'more_than' predicate
        pred1, pred2 = args
        return Function(cfg.lookup['MORE_THAN'], [
            Function(cfg.lookup[pred1.upper()], []), 
            Function(cfg.lookup[pred2.upper()], []),
        ])
    
    elif predicate == 'MORE_OR_EQUAL_THAN':
        # Handle 'more_than' predicate
        pred1, pred2 = args
        return Function(cfg.lookup['MORE_OR_EQUAL_THAN'], [
            Function(cfg.lookup[pred1.upper()], []), 
            Function(cfg.lookup[pred2.upper()], []),
        ])
    
    elif predicate == 'EITHER':
        # Handle 'either' predicate
        n1, n2 = args
        n1 = int(n1)
        n2 = int(n2)
        return Function(cfg.lookup['EITHER'], [
            cfg.lookup[f'constant_{n1}'],
            cfg.lookup[f'constant_{n2}']
        ])
    
    elif predicate == 'LENGTH':
        # Handle 'either' predicate
        n = args
        n1 = int(n[0])
        return Function(cfg.lookup['LENGTH'], [
            cfg.lookup[f'constant_{n1}']
        ])
    else:
        raise ValueError(f"Unsupported predicate: {predicate}")

# zendo_dsl = dsl.DSL(zendo.semantics, zendo.primitive_types, None)
# type_request = Arrow(List(zendo.PIECE), BOOL)
# cfg = zendo_dsl.DSL_to_CFG(
#         type_request, max_program_depth=5)
# # Example Prolog queries
# prolog_query1 = "(AT_LEAST_1 3 IS_RED)"
# prolog_query2 = "(AND (AT_LEAST_1 1 IS_PYRAMID) (AT_LEAST_1 3 IS_RED))"
# prolog_query3 = "(AND (AT_LEAST_1 1 IS_PYRAMID) (OR (AT_LEAST_1 1 IS_WEDGE) (AT_LEAST_INTERACTION 1 (ON_TOP_OF IS_BLOCK IS_BLOCK))))"
# prolog_query4 = "(AND (EXACTLY_1 1 IS_PYRAMID) (AT_LEAST_1 3 IS_RED))"

# # Convert them to DSL
# converted_query1 = convert_string_to_dsl(prolog_query1, cfg)
# converted_query2 = convert_string_to_dsl(prolog_query2, cfg)
# converted_query3 = convert_string_to_dsl(prolog_query3, cfg)
# converted_query4 = convert_string_to_dsl(prolog_query4, cfg)

# # Print the results
# print("Converted Query 1:", converted_query1)
# print("Converted Query 2:", converted_query2)
# print("Converted Query 3:", converted_query3)
# print("Converted Query 4:", converted_query4)
