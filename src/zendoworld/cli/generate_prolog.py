"""Run a Prolog generation query in an isolated Python process."""
import json
import sys


def main():
    from zendoworld.rules.rules import generate_prolog_structure
    result = generate_prolog_structure(int(sys.argv[1]), sys.argv[2], sys.argv[3])
    if not result:
        print("No results returned from generate_prolog_structure", file=sys.stderr)
        raise SystemExit(1)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
