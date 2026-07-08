from program import Function, BasicPrimitive

def dsl_to_prolog(func: Function) -> str:
    """Recursively convert a DSL Function into a Prolog rule string."""
    
    atomic_map = {
        'IS_RED': 'red',
        'IS_BLUE': 'blue',
        'IS_YELLOW': 'yellow',
        'IS_BLOCK': 'block',
        'IS_WEDGE': 'wedge',
        'IS_PYRAMID': 'pyramid',
        'IS_UPRIGHT': 'upright',
        'IS_FLAT': 'flat',
        'IS_UPSIDE_DOWN': 'upside_down',
        'IS_VERTICAL': 'vertical',
        'IS_CHEESECAKE': 'cheesecake',
        'IS_DOORSTOP': 'doorstop',
        'IS_GROUNDED': 'grounded',
        'IS_UNGROUNDED': 'ungrounded',
    }

    def unwrap(arg):
        if isinstance(arg, Function):
            return dsl_to_prolog(arg)
        elif isinstance(arg, BasicPrimitive):
            if arg.primitive in atomic_map:
                return atomic_map[arg.primitive]
            if arg.primitive == "EVEN":
                return "even_number_of(Structure)"
            if arg.primitive == "ODD":
                return "odd_number_of(Structure)"
            if arg.primitive.startswith("constant_"):
                return arg.primitive.replace("constant_", "")
            if arg.primitive in ['ALL_THREE_SHAPES', 'ALL_THREE_COLORS']:
                return f"{arg.primitive.lower()}(Structure)"
            return arg.primitive
        return str(arg)

    head = func.function
    args = func.arguments

    if head.primitive in atomic_map:
        return atomic_map[head.primitive]

    if head.primitive.startswith("constant_"):
        return head.primitive.replace("constant_", "")

    if head.primitive == 'SAME_AMOUNT':
        pred1 = unwrap(args[0])
        pred2 = unwrap(args[1])
        return f"same_amount({pred1}, {pred2}, Structure)"
    elif head.primitive == 'AT_LEAST_1':
        count = unwrap(args[0])
        pred = unwrap(args[1])
        return f"at_least({pred}, {count}, Structure)"
    elif head.primitive == 'EXACTLY_1':
        count = unwrap(args[0])
        pred = unwrap(args[1])
        return f"exactly({pred}, {count}, Structure)"
    elif head.primitive == 'AT_LEAST_2':
        count = unwrap(args[0])
        pred1 = unwrap(args[1])
        pred2 = unwrap(args[2])
        if pred1 == 'grounded':
            return f"at_least_interaction({pred2}, grounded, {count}, Structure)"
        if pred2 == 'grounded':
            return f"at_least_interaction({pred1}, grounded, {count}, Structure)"
        if pred1 == 'ungrounded':
            return f"at_least_interaction({pred2}, ungrounded, {count}, Structure)"
        if pred2 == 'ungrounded':
            return f"at_least_interaction({pred1}, ungrounded, {count}, Structure)"
        return f"at_least({pred1}, {pred2}, {count}, Structure)"
    elif head.primitive == 'EXACTLY_2':
        count = unwrap(args[0])
        pred1 = unwrap(args[1])
        pred2 = unwrap(args[2])
        if pred1 == 'grounded':
            return f"exactly_interaction({pred2}, grounded, {count}, Structure)"
        if pred2 == 'grounded':
            return f"exactly_interaction({pred1}, grounded, {count}, Structure)"
        if pred1 == 'ungrounded':
            return f"exactly_interaction({pred2}, ungrounded, {count}, Structure)"
        if pred2 == 'ungrounded':
            return f"exactly_interaction({pred1}, ungrounded, {count}, Structure)"
        return f"exactly({pred1}, {pred2}, {count}, Structure)"


    if head.primitive == 'ALL_1':
        pred = unwrap(args[0])
        return f"all({pred}, Structure)"
    if head.primitive == 'ALL_2':
        pred1 = unwrap(args[0])
        pred2 = unwrap(args[1])
        return f"all({pred1}, {pred2}, Structure)"

    if head.primitive in ['EVEN', 'ODD']:
        return f"{head.primitive.lower()}_number_of(Structure)"
    if head.primitive in ['ALL_THREE_SHAPES', 'ALL_THREE_COLORS']:
        return f"{head.primitive.lower()}(Structure)"
    if head.primitive in ['EVEN_1', 'ODD_1']:
        pred = unwrap(args[0])
        return f"{head.primitive.lower().replace('_1', '_number_of')}({pred}, Structure)"
    if head.primitive == 'ZERO_1':
        pred = unwrap(args[0])
        return f"{head.primitive.lower().replace('_1', '')}({pred}, Structure)"
    if head.primitive == 'ZERO_2':
        pred1 = unwrap(args[0])
        pred2 = unwrap(args[1])
        return f"{head.primitive.lower().replace('_2', '')}({pred1}, {pred2}, Structure)"
    if head.primitive == 'EXCLUSIVELY':
        pred = unwrap(args[0])
        return f"{head.primitive.lower()}({pred}, Structure)"
    if head.primitive in ['EVEN_2', 'ODD_2']:
        pred1 = unwrap(args[0])
        pred2 = unwrap(args[1])
        if pred1 == 'grounded':
            return f"{head.primitive.lower().replace('_2', '_number_of_interaction')}({pred2}, grounded, Structure)"
        if pred2 == 'grounded':
            return f"{head.primitive.lower().replace('_2', '_number_of_interaction')}({pred1}, grounded, Structure)"
        if pred1 == 'ungrounded':
            return f"{head.primitive.lower().replace('_2', '_number_of_interaction')}({pred2}, ungrounded, Structure)"
        if pred2 == 'ungrounded':
            return f"{head.primitive.lower().replace('_2', '_number_of_interaction')}({pred1}, ungrounded, Structure)"
        return f"{head.primitive.lower().replace('_2', '_number_of')}({pred1}, {pred2}, Structure)"

    if head.primitive in ['AND', 'OR']:
        left = unwrap(args[0])
        right = unwrap(args[1])
        return f"{head.primitive.lower()}([{left}, {right}])"

    if head.primitive in ['EITHER', 'EITHER_OR']:
        n1 = unwrap(args[0])
        n2 = unwrap(args[1])
        return f"either_or({n1}, {n2}, Structure)"

    if head.primitive == 'MORE_THAN':
        p1 = unwrap(args[0])
        p2 = unwrap(args[1])
        return f"more_than({p1}, {p2}, Structure)"
    
    if head.primitive == "LENGTH":
        pred = unwrap(args[0])
        return f"length_({pred}, Structure)"

    if head.primitive in ['AT_LEAST_INTERACTION', 'EXACTLY_INTERACTION', 'EVEN_INTERACTION', 'ODD_INTERACTION']:
        has_count = head.primitive.startswith("AT_LEAST") or head.primitive.startswith("EXACTLY")
        count = unwrap(args[0]) if has_count else None
        interaction_func = args[1 if has_count else 0]

        inter_name = interaction_func.function.primitive.lower()
        pred1 = unwrap(interaction_func.arguments[0])
        pred2 = unwrap(interaction_func.arguments[1])

        if head.primitive in ['EVEN_INTERACTION', 'ODD_INTERACTION']:
            func_name = f"{head.primitive.lower().replace('_interaction', '_number_of_interaction')}"
        else:
            func_name = head.primitive.lower()

        if has_count:
            return f"{func_name}({pred1}, {pred2}, {inter_name}, {count}, Structure)"
        else:
            return f"{func_name}({pred1}, {pred2}, {inter_name}, Structure)"
    raise ValueError(f"Unsupported primitive in DSL: {head.primitive}")
