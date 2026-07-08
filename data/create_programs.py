import re
from program import Function

# Define function to parse the Prolog query
def parse_prolog_query(query):
    """
    Parse a Prolog-like query, extracting the predicate and arguments.
    Example input: "at_least(yellow, pyramid, 1, Structure)"
    """
    query = query.strip()
    match = re.match(r'(\w+)\((.*)\)', query)
    if not match:
        raise ValueError(f"Invalid query format: {query}")
    
    predicate = match.group(1)
    args_str = match.group(2)
    
    args = []
    paren_depth = 0
    current_arg = ""
    
    for char in args_str:
        if char == '(':
            paren_depth += 1
            current_arg += char
        elif char == ')':
            paren_depth -= 1
            current_arg += char
        elif char == ',' and paren_depth == 0:
            args.append(current_arg.strip())
            current_arg = ""
        else:
            current_arg += char
    
    if current_arg:
        args.append(current_arg.strip())

    return predicate, args

def remove_generate_valid_structure(query):
    """Remove the 'generate_valid_structure' or 'generate_invalid_structure' wrapper if present."""
    pattern = r'^"?generate_(valid|invalid)_structure\(\s*\[(.*)\]\s*,\s*Structure\s*\)"?$'
    match = re.match(pattern, query.strip())
    
    if match:
        return match.group(2).strip()
    else:
        print(f"Warning: Query '{query}' does not match expected format. Returning as is.")
        return query

def convert_prolog_to_dsl(prolog_query, cfg):
    """
    Convert a Prolog query to the corresponding DSL function.
    Example: at_least(yellow, pyramid, 1, Structure) -> semantics (semantics['IS_YELLOW'], semantics['IS_PYRAMID'])
    """
    predicate, args = parse_prolog_query(prolog_query)
    
    if predicate == 'all_three_shapes':
        return Function(cfg.lookup['ALL_THREE_SHAPES'], [])
    if predicate == 'all_three_colors':
        return Function(cfg.lookup['ALL_THREE_COLORS'], [])
    if predicate == 'at_least':
        if len(args) == 4:
            pred1, pred2, count, _ = args
            count = int(count)
            return Function(cfg.lookup['AT_LEAST_2'], [
                cfg.lookup[f'constant_{count}'],
                Function(cfg.lookup[f'IS_{pred1.upper()}'], []), 
                Function(cfg.lookup[f'IS_{pred2.upper()}'], []),
            ])
        elif len(args) == 3:
            pred1, count, _ = args
            count = int(count)
            return Function(cfg.lookup['AT_LEAST_1'], [
                cfg.lookup[f'constant_{count}'],
                Function(cfg.lookup[f'IS_{pred1.upper()}'], []),
            ])
        else:
            raise ValueError(f"Invalid number of arguments for at_least: {len(args)}")
    elif predicate == 'same_amount':
        if len(args) == 3:
            pred1, pred2, _ = args
            return Function(cfg.lookup['SAME_AMOUNT'], [
                Function(cfg.lookup[f'IS_{pred1.upper()}'], []), 
                Function(cfg.lookup[f'IS_{pred2.upper()}'], []),
            ])
        else:
            raise ValueError(f"Invalid number of arguments for same_amount: {len(args)}")
    elif predicate == 'exactly':
        if len(args) == 4:
            pred1, pred2, count, _ = args
            count = int(count)
            return Function(cfg.lookup['EXACTLY_2'], [
                cfg.lookup[f'constant_{count}'],
                Function(cfg.lookup[f'IS_{pred1.upper()}'], []), 
                Function(cfg.lookup[f'IS_{pred2.upper()}'], []),
            ])
        elif len(args) == 3:
            pred1, count, _ = args
            count = int(count)
            return Function(cfg.lookup['EXACTLY_1'], [
                cfg.lookup[f'constant_{count}'],
                Function(cfg.lookup[f'IS_{pred1.upper()}'], []), 
            ])
        else:
            raise ValueError(f"Invalid number of arguments for exactly: {len(args)}")

    elif predicate == 'zero':
        if len(args) == 2:
            pred, _ = args
            return Function(cfg.lookup['ZERO_1'], [
                Function(cfg.lookup[f'IS_{pred.upper()}'], []),
            ])
        elif len(args) == 3:
            pred1, pred2, _ = args
            return Function(cfg.lookup['ZERO_2'], [
                Function(cfg.lookup[f'IS_{pred1.upper()}'], []), 
                Function(cfg.lookup[f'IS_{pred2.upper()}'], []), 
            ])
        else:
            raise ValueError(f"Invalid number of arguments for zero: {len(args)}")

    elif predicate == 'exclusively':
        if len(args) == 2:
            pred, _ = args
            return Function(cfg.lookup['EXCLUSIVELY'], [ 
                Function(cfg.lookup[f'IS_{pred.upper()}'], []),
            ])
        else:
            raise ValueError(f"Invalid number of arguments for exclusively: {len(args)}")

    elif predicate == 'and':
        subquery0 = args[0].strip('[]')
        subquery1 = args[1].strip('[]')
        rule1 = convert_prolog_to_dsl(subquery0, cfg)
        rule2 = convert_prolog_to_dsl(subquery1, cfg)
        return Function(cfg.lookup['AND'], [rule1, rule2])
    
    elif predicate == 'or':
        subquery0 = args[0].strip('[]')
        subquery1 = args[1].strip('[]')
        rule1 = convert_prolog_to_dsl(subquery0, cfg)
        rule2 = convert_prolog_to_dsl(subquery1, cfg)
        return Function(cfg.lookup['OR'], [rule1, rule2])
    
    elif predicate == 'either_or':
        n1, n2, _ = args
        n1 = int(n1)
        n2 = int(n2)
        return Function(cfg.lookup['EITHER_OR'], [cfg.lookup[f'constant_{n1}'], cfg.lookup[f'constant_{n2}']])
    
    elif predicate == 'odd_number_of':
        if len(args) == 1:
            return Function(cfg.lookup['ODD'], [])
        if len(args) == 2:
            attr, _ = args
            return Function(cfg.lookup['ODD_1'], [Function(cfg.lookup[f'IS_{attr.upper()}'], [])])
        elif len(args) == 3:
            pred1, pred2, _ = args
            return Function(cfg.lookup['ODD_2'], [
                    Function(cfg.lookup[f'IS_{pred1.upper()}'], []), 
                    Function(cfg.lookup[f'IS_{pred2.upper()}'], []),
                ])
        else:
            raise ValueError(f"Invalid number of arguments for odd_number_of: {len(args)}")
    
    elif predicate == 'even_number_of':
        if len(args) == 1:
            return Function(cfg.lookup['EVEN'], [])
        if len(args) == 2:
            attr, _ = args
            return Function(cfg.lookup['EVEN_1'], [Function(cfg.lookup[f'IS_{attr.upper()}'], [])])
        elif len(args) == 3:
            pred1, pred2, _ = args
            return Function(cfg.lookup['EVEN_2'], [
                    Function(cfg.lookup[f'IS_{pred1.upper()}'], []), 
                    Function(cfg.lookup[f'IS_{pred2.upper()}'], []),
                ])
        else:
            raise ValueError(f"Invalid number of arguments for even_number_of: {len(args)}")
    
    elif predicate == 'at_least_interaction':
        if len(args) == 4:
            pred, interaction, count, _ = args
            count = int(count)
            return Function(cfg.lookup['AT_LEAST_2'], [
                cfg.lookup[f'constant_{count}'],
                Function(cfg.lookup[f'IS_{interaction.upper()}'], []),
                Function(cfg.lookup[f'IS_{pred.upper()}'], []),
            ])
        if len(args) == 5:
            pred1, pred2, interaction, count, _ = args
            count = int(count)
            return Function(cfg.lookup['AT_LEAST_INTERACTION'], [
                cfg.lookup[f'constant_{count}'],
                Function(cfg.lookup[interaction.upper()],
                [
                    Function(cfg.lookup[f'IS_{pred1.upper()}'], []), 
                    Function(cfg.lookup[f'IS_{pred2.upper()}'], []),
                ])
            ])
        else:
            raise ValueError(f"Invalid number of arguments for at_least_interaction: {len(args)}")
    
    elif predicate == 'exactly_interaction':
        if len(args) == 4:
            pred, interaction, count, _ = args
            count = int(count)
            return Function(cfg.lookup['EXACTLY_2'], [
                cfg.lookup[f'constant_{count}'],
                Function(cfg.lookup[f'IS_{interaction.upper()}'], []),
                Function(cfg.lookup[f'IS_{pred.upper()}'], []),
            ])
        if len(args) == 5:
            pred1, pred2, interaction, count, _ = args
            count = int(count)
            return Function(cfg.lookup['EXACTLY_INTERACTION'], [
                cfg.lookup[f'constant_{count}'],
                Function(cfg.lookup[interaction.upper()],
                [
                    Function(cfg.lookup[f'IS_{pred1.upper()}'], []), 
                    Function(cfg.lookup[f'IS_{pred2.upper()}'], []),
                ])
            ])
        else:
            raise ValueError(f"Invalid number of arguments for exactly_interaction: {len(args)}")
        
    elif predicate == 'odd_number_of_interaction':
        if len(args) == 3:
            pred, interaction, _ = args
            return Function(cfg.lookup['ODD_2'], [
                Function(cfg.lookup[f'IS_{interaction.upper()}'], []),
                Function(cfg.lookup[f'IS_{pred.upper()}'], []),
            ])
        if len(args) == 4:
            pred1, pred2, interaction, _ = args
            return Function(cfg.lookup['ODD_INTERACTION'], [
                Function(cfg.lookup[interaction.upper()],
                [
                    Function(cfg.lookup[f'IS_{pred1.upper()}'], []), 
                    Function(cfg.lookup[f'IS_{pred2.upper()}'], []),
                ])
            ])
        else:
            raise ValueError(f"Invalid number of arguments for odd_number_of_interaction: {len(args)}")
        
    elif predicate == 'even_number_of_interaction':
        if len(args) == 3:
            pred, interaction, _ = args
            return Function(cfg.lookup['EVEN_2'], [
                Function(cfg.lookup[f'IS_{interaction.upper()}'], []),
                Function(cfg.lookup[f'IS_{pred.upper()}'], []),
            ])
        if len(args) == 4:
            pred1, pred2, interaction, _ = args
            return Function(cfg.lookup['EVEN_INTERACTION'], [
                Function(cfg.lookup[interaction.upper()],
                [
                    Function(cfg.lookup[f'IS_{pred1.upper()}'], []), 
                    Function(cfg.lookup[f'IS_{pred2.upper()}'], []),
                ])
            ])
        else:
            raise ValueError(f"Invalid number of arguments for even_number_of_interaction: {len(args)}")
    
    elif predicate == 'more_than':
        pred1, pred2, _ = args
        return Function(cfg.lookup['MORE_THAN'], [
            Function(cfg.lookup[f'IS_{pred1.upper()}'], []), 
            Function(cfg.lookup[f'IS_{pred2.upper()}'], []),
        ])
    
    elif predicate == 'either':
        n1, n2, _ = args
        n1 = int(n1)
        n2 = int(n2)
        return Function(cfg.lookup['EITHER'], [
            cfg.lookup[f'constant_{n1}'],
            cfg.lookup[f'constant_{n2}']
        ])
    
    else:
        raise ValueError(f"Unsupported predicate: {predicate}")

# # Example Prolog queries
# prolog_query1 = "at_least(yellow, pyramid, 1, Structure)"
# prolog_query2 = "exactly(red, block, 2, Structure)"
# prolog_query3 = "and([at_least(yellow, pyramid, 1, Structure), exactly(red, block, 2, Structure)])"
# prolog_query4 = "more_than(upside_down, upright, Structure)"
# prolog_query5 = "either_or(2, 1, Structure)"
# prolog_query6 = "exactly_interaction(blue, block, touching, 1, Structure)"
# prolog_query7 = "even_number_of(red, Structure)"
# prolog_query8 = "odd_number_of_interaction(upside_down, red, touching, Structure)"
# prolog_query9 = "even_number_of_interaction(blue, yellow, pointing, Structure)"
# prolog_query10 = "even_number_of_interaction(wedge, yellow, on_top_of, Structure)"
# prolog_query11 = "or([even_number_of(upside_down, Structure), more_than(wedge, block, Structure)])"
# prolog_query12 = "either_or(2, 1, Structure)"

# # Convert them to DSL
# converted_query1 = convert_prolog_to_dsl(prolog_query1)
# converted_query2 = convert_prolog_to_dsl(prolog_query2)
# converted_query3 = convert_prolog_to_dsl(prolog_query3)
# converted_query4 = convert_prolog_to_dsl(prolog_query4)
# converted_query5 = convert_prolog_to_dsl(prolog_query5)
# converted_query6 = convert_prolog_to_dsl(prolog_query6)
# converted_query7 = convert_prolog_to_dsl(prolog_query7)
# converted_query8 = convert_prolog_to_dsl(prolog_query8)
# converted_query9 = convert_prolog_to_dsl(prolog_query9)
# converted_query10 = convert_prolog_to_dsl(prolog_query10)
# converted_query11 = convert_prolog_to_dsl(prolog_query11)
# converted_query12 = convert_prolog_to_dsl(prolog_query12)

# # Print the results
# print("Converted Query 1:", converted_query1)
# print("Converted Query 2:", converted_query2)
# print("Converted Query 3:", converted_query3)
# print("Converted Query 4:", converted_query4)
# print("Converted Query 5:", converted_query5)
# print("Converted Query 6:", converted_query6)
# print("Converted Query 7:", converted_query7)
# print("Converted Query 8:", converted_query8)
# print("Converted Query 9:", converted_query9)
# print("Converted Query 10:", converted_query10)
# print("Converted Query 11:", converted_query11)
# print("Converted Query 12:", converted_query12)
