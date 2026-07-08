# This file contains code derived from:
# - https://github.com/CapArrow/zendo_game_dataset_generator
import sys, os, random, json, re
from dataclasses import dataclass, field
from pyswip import Prolog
from hashlib import sha256
from pathlib import Path


@dataclass
class PlaceholderTemplate:
    """
    A placeholder template describes a text to be inserted in the place of its placeholder_id.
    It can itself contain placeholders, which is why it is split up into tokens.
    """

    @dataclass(frozen=True)
    class Token:
        string: str
        is_placeholder: bool

    placeholder_id: str
    template: str
    placeholders: set[str]
    tokens: list[Token]
    prolog: list[str]
    orientations: list[str]


@dataclass
class Placeholder:
    """
    A placeholder is to be replaced by any of its placeholder templates stored in all_templates.
    """

    categories: dict[str, list[PlaceholderTemplate]]
    templates_of_start_match: dict[str, list[PlaceholderTemplate]]
    all_templates: list[PlaceholderTemplate]
    start_pattern: re.Pattern[str]


@dataclass
class Rules:
    # Maps QUANTITY, OPERATION, INTERACTION, ORIENTATION, COLOR, etc. to their respective placeholder rules.
    placeholders: dict[str, Placeholder]

    # Maps the first token string of a template to itself
    templates_of_start_match: dict[str, list[PlaceholderTemplate]]
    templates_start_pattern: re.Pattern[str]


@dataclass
class RuleNode:
    template: PlaceholderTemplate
    children: list["RuleNode"]


@dataclass
class RuleParser:
    rules: Rules
    depth: int
    print_tree: bool


@dataclass
class TemplateGenerator:
    rules: Rules
    depth: int
    used_templates: dict[str, list[PlaceholderTemplate]]
    two_random_steps: bool = False
    print_tree: bool = False


# MARK: Exceptions

def raise_unrecognized_placeholder(placeholder_id: str, recognized_placeholders: set[str]):
    """
    Raises an exception saying that placeholder_id is not recognized because it is not part of recognized_placeholders.

    :param placeholder_id: The placeholder identifier that is not recognized.
    :param recognized_placeholders: A set of valid placeholder identifiers.
    :raises Exception: That placeholder_id is not found in the recognized_placeholders.
    """

    recognized_placeholders_string = ", ".join(list(recognized_placeholders))
    raise Exception(
        f"Placeholder '{placeholder_id}' is not recognized as it is not part of the rules placeholders ({recognized_placeholders_string}).")


# MARK: General

def make_placeholder_template(placeholder_id: str, template_text: str, prolog_list: list[str],
                              orientations: list[str]) -> PlaceholderTemplate:
    """
    Creates a placeholder template by tokenizing the given template_text.
    Placeholders in the text must be surrounded by curly braces: {IDENTIFIER}.

    :param placeholder_id: The identifier for the placeholder template.
    :param template_text: The text containing placeholders enclosed in curly braces.
    :param prolog_list: The equivalent prolog identifier of this placeholder template in a list. Should contain a single or no element.
    :param orientations: A list of possible orientations for the placeholder template. Only applicable if of type SHAPE.
    :return: A PlaceholderTemplate object with tokenized text and extracted placeholders.
    """

    tokens: list[PlaceholderTemplate.Token] = []
    placeholders: set[str] = set()

    last_end_index = 0
    for placeholder_match in re.finditer(r"\{([^\{\}]*)\}", template_text):
        match_start_index = placeholder_match.start()
        if last_end_index < match_start_index:
            tokens.append(PlaceholderTemplate.Token(template_text[last_end_index:match_start_index], False))

        placeholder = placeholder_match.group(1)
        placeholders.add(placeholder)
        tokens.append(PlaceholderTemplate.Token(placeholder, True))
        last_end_index = placeholder_match.end()

    # Add last template_text part or empty string if there are no tokens (like for the none operation, which doesn't have a text)
    if last_end_index < len(template_text) or len(tokens) == 0:
        tokens.append(PlaceholderTemplate.Token(template_text[last_end_index:], False))

    return PlaceholderTemplate(placeholder_id, template_text, placeholders, tokens, prolog_list, orientations)


def load_json_rules(filename) -> Rules | None:
    """
    Loads placeholder rules from a JSON file and constructs a Rules object.

    :param filename: The path to the JSON file containing the rules.
    :return: A Rules object containing parsed placeholder rules, or None if loading fails.
    """

    with open(filename) as f:
        data = json.load(f)
        placeholders: dict[str, Placeholder] = {}

        def make_placeholder_template_list(identifier: str, templates: list[dict] | list[str]):
            """
            Creates a list of PlaceholderTemplate objects from a list of template definitions.

            :param identifier: The identifier for the placeholder category.
            :param templates: A list of template definitions, either as dictionaries or strings.
            :return: A list of PlaceholderTemplate objects.
            """

            result = []
            placeholder_pattern = re.compile(r"{([A-Z]+(?:\|[A-Z]+)*)}")
            placeholder_empty_pattern = re.compile(r"{}")
            for template in templates:
                if isinstance(template, dict) and "group" in template and "includes" in template:
                    continue  # Group info is handled outside here
                # NOTE(kilian): Allow for short list where "template" is omitted and only the template values are stored
                template_prolog = template["prolog"] if type(template) == dict and "prolog" in template else []
                template_orientations = template["orientations"] if type(
                    template) == dict and "orientations" in template else []
                template_string = template['template'] if type(template) == dict else str(template)

                # NOTE(kilian): Compute list of placeholders contained in the template
                template_placeholders = []
                while True:
                    placeholder_match = re.search(placeholder_pattern, template_string)
                    if not placeholder_match:
                        break
                    template_placeholder_multiple = placeholder_match.group(1)
                    template_placeholders.append(template_placeholder_multiple.split("|"))
                    template_string = template_string[:placeholder_match.start(1)] + template_string[
                                                                                     placeholder_match.end(1):]

                def configure_placeholders(result, template_string, i, template_placeholders, selected_placeholders):
                    """
                    Recursively configures placeholders by replacing multiple options with actual values.

                    :param result: List to store generated PlaceholderTemplate objects.
                    :param template_string: The template string being processed.
                    :param i: The current index in template_placeholders.
                    :param template_placeholders: A list of possible placeholder values.
                    :param selected_placeholders: Selected placeholders for replacement.
                    """

                    if i >= len(template_placeholders):
                        result.append(make_placeholder_template(identifier, template_string, template_prolog,
                                                                template_orientations))
                    else:
                        for p in template_placeholders[i]:
                            selected_placeholders[i] = p
                            replaced_template_string = re.sub(placeholder_empty_pattern, f"{{{p}}}", template_string,
                                                              count=1)
                            configure_placeholders(result, replaced_template_string, i + 1, template_placeholders,
                                                   selected_placeholders)

                selected_placeholders = [0] * len(template_placeholders)
                configure_placeholders(result, template_string, 0, template_placeholders, selected_placeholders)
            return result

        def update_templates_by_match(templates_by_match, all_templates: list[PlaceholderTemplate]):
            """
            Updates a dictionary mapping starting template tokens to their respective templates.

            :param templates_by_match: Dictionary mapping start tokens to placeholder templates.
            :param all_templates: List of all PlaceholderTemplate objects to be indexed.
            """

            for template in all_templates:
                token = template.tokens[0]
                # NOTE(kilian): If the first token is a placeholder, matching the start won't work like for the other templates, so make it match always by using ""
                token_string = "" if token.is_placeholder else token.string
                if token_string != "" and token_string not in templates_by_match:
                    for key in templates_by_match.keys():
                        if key != "" and token_string.startswith(key):
                            token_string = key
                            break
                        if key.startswith(token_string):
                            templates_by_match[token_string] = templates_by_match[key]
                            del templates_by_match[key]
                            break
                    else:
                        templates_by_match[token_string] = []
                templates_by_match[token_string].append(template)

        def make_templates_by_match_pattern(templates_by_match):
            """
            Constructs a regular expression pattern to match starting tokens of templates.

            :param templates_by_match: Dictionary mapping start tokens to placeholder templates.
            :return: A compiled regex pattern for template matching.
            """

            templates_start_patern_string = "|".join([re.escape(key) for key in templates_by_match.keys()])
            return re.compile(f"^(?:{templates_start_patern_string})")

        def make_placeholder(categories: dict[str, list[PlaceholderTemplate]]):
            """
            Creates a Placeholder object from categorized templates.

            :param categories: A dictionary mapping categories to lists of PlaceholderTemplate objects.
            :return: A Placeholder object containing categorized templates.
            """

            all_templates = [t for templates in categories.values() for t in templates]
            placeholder_templates_of_start_match = {"": []}
            update_templates_by_match(placeholder_templates_of_start_match, all_templates)
            placeholder_start_pattern = make_templates_by_match_pattern(placeholder_templates_of_start_match)
            return Placeholder(categories, placeholder_templates_of_start_match, all_templates,
                               placeholder_start_pattern)

        for placeholder_identifier, placeholder_values in data.items():
            placeholder_type = type(placeholder_values)
            # NOTE(kilian): Differentiate between categorized templates (like for QUANTITY with "at least") and plain lists of templates
            if placeholder_type == dict:
                placeholders[placeholder_identifier] = make_placeholder(
                    {value: make_placeholder_template_list(placeholder_identifier, templates) for value, templates in
                     placeholder_values.items()})
            else:
                assert (placeholder_type == list)
                placeholders[placeholder_identifier] = make_placeholder(
                    {"": make_placeholder_template_list(placeholder_identifier, placeholder_values)})

        # NOTE(kilian): Group all templates and per placeholder by starting pattern to be able to match more easily later
        templates_of_match = {"": []}
        for placeholder in placeholders.values():
            update_templates_by_match(templates_of_match, placeholder.all_templates)
        rules = Rules(placeholders=placeholders, templates_of_start_match=templates_of_match,
                      templates_start_pattern=make_templates_by_match_pattern(templates_of_match))
        return rules
    return None


def print_rules(rules: Rules):
    """
    Prints all placeholder rules in a structured format.

    :param rules: A Rules object containing placeholders and their templates.
    """

    for placeholder_identifier, placeholder in rules.placeholders.items():
        for placeholder_value, placeholder_templates in placeholder.categories.items():
            tabs_str = "\t"
            if placeholder_value != "":
                tabs_str += "\t"


def template_from_text(rules: Rules, text: str, placeholders_naked: bool = True) -> PlaceholderTemplate:
    """
    Creates a placeholder template from plain text. Only placeholder identifiers of the given rules are recognized.

    :param rules: A Rules object containing recognized placeholders.
    :param text: The input text containing placeholder identifiers.
    :param placeholders_naked: If True, placeholder identifiers are expected without curly braces and will be wrapped.
    :return: A PlaceholderTemplate object created from the text.
    :raises Exception: If any unrecognized placeholder is found in the text.
    """

    template = text[:]
    recognized_placeholders: set[str] = set(rules.placeholders.keys())
    if placeholders_naked:
        for placeholder_identifier in recognized_placeholders:
            if placeholder_identifier in text:
                template = template.replace(placeholder_identifier, f"{{{placeholder_identifier}}}")

    result = make_placeholder_template("CUSTOM", template, [], [])
    for placeholder in result.placeholders:
        if placeholder not in recognized_placeholders:
            raise_unrecognized_placeholder(placeholder, recognized_placeholders)
    return result


# MARK: Generating

def random_placeholder_template(rules: Rules, placeholder_id: str, two_steps: bool = False,
                                used_templates: dict[str, list[PlaceholderTemplate]] | None = None,
                                allowed_orientations: list[PlaceholderTemplate] | None = None,
                                no_interaction_when_operator_used: bool = True) -> PlaceholderTemplate:
    """
    Randomly selects a placeholder template from the rules based on the given placeholder_id.
    Ensures that previously used templates (if provided) are not reused.

    :param rules: A Rules object containing placeholder definitions.
    :param placeholder_id: The identifier of the placeholder category to select from.
    :param two_steps: If True, selects a category first, then a template from that category.
    If False, selects from the templates of all categories uniformly.
    :param used_templates: A dictionary tracking already used templates to avoid repetition.
    :param allowed_orientations: A list of specific orientation templates to be considered, if applicable.
    :param no_interaction_when_operator_used: If True, avoids interaction templates if an operator was used previously.
    :return: A randomly chosen PlaceholderTemplate object.
    :raises Exception: If no valid templates are available.
    """

    if placeholder_id not in rules.placeholders:
        raise_unrecognized_placeholder(placeholder_id, rules.placeholders.keys())

    # NOTE(kilian): First, pick placeholder_templates
    if placeholder_id == "ORIENTATION" and allowed_orientations:
        placeholder_templates = allowed_orientations

        if len(placeholder_templates) == 0:
            raise Exception("No templates specified as allowed orientations.")
    elif two_steps:
        placeholder = rules.placeholders[placeholder_id]
        if len(placeholder.categories) == 1:
            if "" not in placeholder.categories:
                raise Exception(f"Single category of placeholder '{placeholder_id}' is not \"\".")
            picked_placeholder_category = ""
        else:
            picked_placeholder_category = random.choice(list(placeholder.categories.keys()))

        placeholder_templates = placeholder.categories[picked_placeholder_category]
        if len(placeholder_templates) == 0:
            raise Exception(
                f"No templates for placeholder '{placeholder_id}' in category '{picked_placeholder_category}'.")
    else:
        placeholder_templates = rules.placeholders[placeholder_id].all_templates

        if len(placeholder_templates) == 0:
            raise Exception(f"No templates for placeholder '{placeholder_id}'.")

    # NOTE(kilian): Next, pick one
    picked_template = None
    if placeholder_id == "OPERATION":
        # NOTE(kilian): If any operator has already been used, choose the none operator (pick by choosing the one without placeholders)
        used_operators = used_templates.get(placeholder_id, [])
        if len(used_operators) > 0:
            for template in placeholder_templates:
                if len(template.placeholders) == 0:
                    picked_template = template
    if picked_template is None:
        # NOTE(kilian): Pick any placeholder template that wasn't used already
        # TODO(kilian): Find better to check if two templates are equal? Currently checking if their template texts are equal
        used_placeholder_template_texts = [template.template for template in used_templates.get(placeholder_id, [])]
        remaining_placeholder_templates = [template for template in placeholder_templates if
                                           template.template not in used_placeholder_template_texts]
        # TODO(kilian): Possibly an issue in the future because none is also considered as an operator here?
        if no_interaction_when_operator_used and len(used_templates.get("OPERATION", [])) > 0:
            # NOTE(kilian): Remove any templates that contain interaction placeholders
            remaining_placeholder_templates = [template for template in remaining_placeholder_templates if
                                               "INTERACTION" not in template.placeholders]
        if len(remaining_placeholder_templates) == 0:
            raise Exception(f"Every template of placeholder '{placeholder_id}' was already used.")
        picked_template = random.choice(remaining_placeholder_templates)

    if placeholder_id not in used_templates:
        used_templates[placeholder_id] = []
    used_templates[placeholder_id].append(picked_template)

    return picked_template


def template_to_string_random_recursive(generator: TemplateGenerator, template: PlaceholderTemplate) -> str:
    """
    Recursively generates a random string representation of a template by replacing its placeholders with corresponding values.

    :param generator: A TemplateGenerator object containing the rules and used templates.
    :param template: The PlaceholderTemplate object to be processed.
    :return: A string with placeholders replaced by randomly selected values.
    """

    if generator.print_tree:
        tabs_string = '\t' * generator.depth
        debug(f"{tabs_string}{template.template}")

    shape_replacements = []
    # NOTE(kilian): Pre generate all shape placeholders such that orientation restrictions can be applied
    for token in template.tokens:
        if token.is_placeholder and token.string == "SHAPE":
            try:
                shape_replacement = random_placeholder_template(generator.rules, token.string,
                                                                generator.two_random_steps, generator.used_templates)
            except Exception as e:
                shape_replacement = f"[Error: {str(e)}]"
            shape_replacements.append(shape_replacement)

    def compute_allowed_orientations(shape_index: int) -> list[PlaceholderTemplate] | None:
        """
        Determines the allowed orientation templates for a given shape index.

        :param shape_index: The index of the shape in the list of shape replacements.
        :return: A list of allowed orientation templates or None if unrestricted.
        """

        allowed_orientations = []
        if len(shape_replacements) == 0:
            return None

        if shape_index >= len(shape_replacements):
            # NOTE(kilian): This shouldn't happen
            assert False

        shape_replacement = shape_replacements[shape_index]

        # TODO(kilian): This is quite bad, should turn list of orientation strings into list of templates when loading the rules
        if isinstance(shape_replacement, PlaceholderTemplate):
            for orientation_string in shape_replacement.orientations:
                if orientation_string in generator.rules.placeholders["ORIENTATION"].templates_of_start_match:
                    orientation_templates = generator.rules.placeholders["ORIENTATION"].templates_of_start_match[
                        orientation_string]
                    assert (len(orientation_templates) == 1)
                    allowed_orientations.append(orientation_templates[0])
        return allowed_orientations

    next_shape_index = 0

    result = ""
    token_placeholder_count = 0
    for i, token in enumerate(template.tokens):
        if token.is_placeholder:
            if token.string == "SHAPE":
                replacement = shape_replacements[next_shape_index]
                if isinstance(replacement, PlaceholderTemplate):
                    result += replacement.template
                else:
                    result += replacement
                next_shape_index += 1
            else:
                allowed_orientations = None
                if i + 1 < len(template.tokens):
                    next_token = template.tokens[i + 1]
                    if next_token.is_placeholder and next_token.string == "SHAPE":
                        allowed_orientations = compute_allowed_orientations(next_shape_index)
                try:
                    replacement = random_placeholder_template(generator.rules, token.string, generator.two_random_steps,
                                                              generator.used_templates, allowed_orientations)
                    generator.depth += 1
                    replacement = template_to_string_random_recursive(generator, replacement)
                    generator.depth -= 1
                    result += replacement
                except Exception as e:
                    result += f"[Error: {str(e)}]"

            token_placeholder_count += 1
        else:
            result += token.string
    return result


def random_rule(rules: Rules, starting_template: PlaceholderTemplate, two_random_steps: bool = False,
                print_tree: bool = False) -> str:
    """
    Generates a random rule as a string by recursively replacing placeholders in a starting template.

    :param rules: A Rules object containing placeholder definitions.
    :param starting_template: The initial PlaceholderTemplate from which the rule is generated.
    :param two_random_steps: If True, selects placeholder templates in two steps (category first, then template).
    :param print_tree: If True, prints the generation process as a tree structure.
    :return: A string representation of the randomly generated rule.
    """

    generator = TemplateGenerator(rules, depth=0, used_templates={}, two_random_steps=two_random_steps,
                                  print_tree=print_tree)
    string = template_to_string_random_recursive(generator, starting_template)
    return string


# MARK: Parsing

def parse_rule_text_match(parser: RuleParser, rule: str, template: PlaceholderTemplate) -> tuple[RuleNode, str] | str:
    """
    Matches a given rule text against a template recursively and constructs a corresponding RuleNode tree.

    :param parser: A RuleParser object that holds parsing settings and rules.
    :param rule: The rule text to be parsed.
    :param template: The PlaceholderTemplate against which the rule text is matched.
    :return: A tuple containing the parsed RuleNode and the remaining unmatched rule text,
             or an error message specifying why the template does not match.
    """

    remaining_rule = rule
    nodes = []
    for i, token in enumerate(template.tokens):
        if parser.print_tree:
            tabs_string = '\t' * parser.depth
        if token.is_placeholder:
            token_placeholder = parser.rules.placeholders[token.string]

            # NOTE(kilian): Use longest match (otherwise it will just match none "" anywhere possible)
            possible_templates: list[PlaceholderTemplate] = []
            for possible_match in re.findall(token_placeholder.start_pattern, remaining_rule):
                possible_templates += token_placeholder.templates_of_start_match[possible_match]

            longest_match_node = None
            longest_match_length = 0
            longest_match_remaining_rule = ""
            for test_template in possible_templates:
                parser.depth += 1
                parse_match = parse_rule_text_match(parser, remaining_rule, test_template)
                parser.depth -= 1
                if isinstance(parse_match, str):
                    pass
                else:
                    parse_match, new_remaining_rule = parse_match
                    match_length = len(remaining_rule) - len(new_remaining_rule)
                    if match_length >= longest_match_length:
                        longest_match_node = parse_match
                        longest_match_length = match_length
                        longest_match_remaining_rule = new_remaining_rule

            if not longest_match_node:
                return "ERROR"
            nodes.append(longest_match_node)
            remaining_rule = longest_match_remaining_rule
        else:
            m = re.match(re.escape(token.string), remaining_rule)
            if not m:
                return f"Expected '{token.string}', found '{remaining_rule[:len(token.string)]}'."
            remaining_rule = remaining_rule[m.end(0):]
    return (RuleNode(template, nodes), remaining_rule)


def parse_rule_text(rules: Rules, rule: str, starting_template: PlaceholderTemplate,
                    debug_print=False) -> RuleNode | None:
    """
    Parses the given rule text and generates a tree of RuleNodes based on a starting template.

    :param rules: A Rules object containing all placeholders and templates.
    :param rule: The rule text to parse.
    :param starting_template: The initial PlaceholderTemplate used for matching the rule text.
    :param debug_print: If True, prints debugging information during the parsing process.
    :return: The root RuleNode of the parsed rule tree, or None if parsing fails.
    """

    parser = RuleParser(rules, depth=0, print_tree=debug_print)
    parse_match = parse_rule_text_match(parser, rule, starting_template)
    if isinstance(parse_match, str):
        return None
    root, remaining_rule = parse_match
    return root


def print_rule_nodes(root: RuleNode):
    """
    Prints the structure of a rule node tree in a readable format.

    :param root: The root RuleNode of the rule tree.
    """

    nodes = [(0, root)]
    depth = 0
    while nodes:
        new_nodes = []
        i = 0
        for (num, node) in nodes:
            for c in node.children:
                new_nodes.append((i, c))
            i += 1
        nodes = new_nodes
        depth += 1


# MARK: Prolog conversion

def rule_to_prolog(root: RuleNode) -> tuple[str, str]:
    """
    Converts a parsed rule tree into corresponding Prolog queries.

    :param root: The root RuleNode representing the parsed rule.
    :return: A tuple containing Prolog queries for valid and invalid structure generation.
    """

    @dataclass
    class PrologCall:
        function_name: str | None = None
        quantity: list[str] = field(default_factory=lambda: [])
        number: list[str] = field(default_factory=lambda: [])
        interaction: list[str] = field(default_factory=lambda: [])
        interaction_name: str | None = None

    def nodes_dfs(ors: list[list[PrologCall]], node: RuleNode):
        """
        Recursively traverses the rule tree to construct Prolog calls.

        :param ors: A list of lists containing PrologCall objects for 'or' and 'and' logic.
        :param node: The current RuleNode being processed.
        """

        # NOTE(kilian): Generates an "or" list of "anded" PrologCall objects
        if node is None:
            return
        for child in node.children:
            if len(child.template.prolog) == 0:
                if node.template.placeholder_id == "QUANTITY":
                    if child.template.placeholder_id == "NUMBER":
                        ors[-1][-1].number.append(child.template.template)
                    else:
                        ors[-1][-1].quantity.append(child.template.template)
                elif node.template.placeholder_id == "INTERACTION":
                    ors[-1][-1].interaction.append(child.template.template)
            else:
                match child.template.placeholder_id:
                    case "INTERACTION":
                        ors[-1][-1].interaction_name = child.template.prolog[0]
                    case "QUANTITY":
                        ors[-1][-1].function_name = child.template.prolog[0]
                    case "OPERATION":
                        if "and" in child.template.template:
                            ors[-1].append(PrologCall())
                        elif "or" in child.template.template:
                            ors.append([PrologCall()])
                nodes_dfs(ors, child)

    prologs = [[PrologCall()]]
    nodes_dfs(prologs, root)

    anded_call_strings = []
    for anded_calls in prologs:
        call_strings = []
        for call in anded_calls:
            # NOTE(kilian): Combine arguments in correct order according to rules.pl
            argument_parts = []
            function_name = call.function_name
            if call.quantity:
                argument_parts.append(", ".join(call.quantity))
            if call.interaction:
                argument_parts.append(", ".join(call.interaction))
            if call.interaction_name:
                argument_parts.append(call.interaction_name)
                if call.interaction_name == "grounded":
                    function_name = function_name.removesuffix("_interaction")
            if call.number:
                argument_parts.append(", ".join(call.number))
            argument_parts.append("Structure")

            call_strings.append(f"{function_name}({', '.join(argument_parts)})")

        if len(call_strings) == 1:
            anded_call_strings.append(call_strings[0])
        else:
            anded_call_strings.append(f"and([{', '.join(call_strings)}])")

    ored_call_string = f"or([{', '.join(anded_call_strings)}])" if len(anded_call_strings) != 1 else anded_call_strings[0]
    return f"generate_valid_structure([{ored_call_string}], Structure)", f"generate_invalid_structure([{ored_call_string}], Structure)"


# MARK: Combined

def rule_text_to_prolog(rules: Rules, rule: str, starting_template: PlaceholderTemplate, debug_print_parse=False,
                        debug_print_nodes=False) -> tuple[str, str]:
    """
    Converts a textual rule into its corresponding Prolog representation.

    :param rules: A Rules object containing placeholder definitions.
    :param rule: The rule text to be converted.
    :param starting_template: The initial PlaceholderTemplate used to parse the rule.
    :param debug_print_parse: If True, enables debug printing for the parsing process.
    :param debug_print_nodes: If True, prints the parsed rule node structure.
    :return: A tuple containing Prolog queries for valid and invalid structure generation.
    """

    root = parse_rule_text(rules, rule, starting_template, debug_print=debug_print_parse)
    if debug_print_nodes:
        print_rule_nodes(root)
    return rule_to_prolog(root)


def generate_rule(rules_json_file='rules/zendo_rules.json'):
    """
    Generates a random rule from a JSON rules file and converts it into a Prolog query.

    :param rules_json_file: The path to the JSON file containing placeholder rules.
    :return: A tuple containing the generated rule text, its Prolog query, and its negative Prolog query.
    """

    rules = load_json_rules(rules_json_file)
    starting_template = template_from_text(rules, "A structure must contain QUANTITY.")

    # Generate random rule and parse it into a prolog query
    rule = random_rule(rules, starting_template, two_random_steps=False, print_tree=False)
    query, n_query = rule_text_to_prolog(rules, rule, starting_template, debug_print_parse=False,
                                         debug_print_nodes=False)
    return rule, query, n_query


def generate_prolog_structure(num_examples, query, prolog_file='rules/rules.pl'):
    """
    Executes a Prolog query to generate structural examples.

    :param num_examples: The number of examples to generate.
    :param query: The Prolog query to execute.
    :param prolog_file: The path to the Prolog file containing the rules.
    :return: A list of generated structures.
    """

    prolog = Prolog()
    prolog.consult(prolog_file)

    # Execute the random queries
    seen_structures = set()
    results = []

    def structure_to_hashable(structure):
        """Convert a Prolog structure to a JSON-serializable hashable string."""
        try:
            return sha256(json.dumps(structure, sort_keys=True).encode()).hexdigest()
        except Exception:
            return str(structure)  # fallback

    attempts = 0
    max_attempts = num_examples * 20  # to avoid infinite loops

    while len(results) < num_examples and attempts < max_attempts:
        attempts += 1
        try:
            prolog_query = prolog.query(query)
            for i, szene in enumerate(prolog_query):
                structure = szene["Structure"]
                hash_key = structure_to_hashable(structure)

                if hash_key not in seen_structures:
                    seen_structures.add(hash_key)
                    results.append(structure)

        except Exception as e:
            print(f"Prolog query failed: {e}")
            continue

    if len(results) < num_examples:
        print(f"⚠️ Warning: Only {len(results)} unique structures could be generated after {attempts} attempts.")

    return results


if __name__ == "__main__":
    """
    Main script execution to generate a rule and corresponding Prolog structures.
    """

    for _ in range(100):
        rule, query, n_query = generate_rule()
        r1 = generate_prolog_structure(1, query)
        r2 = generate_prolog_structure(1, n_query)

        print(len(r1[0]), r1)
