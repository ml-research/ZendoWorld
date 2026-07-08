import sys, json
from rules.rules import generate_prolog_structure

if __name__ == "__main__":
    n = int(sys.argv[1])
    query = sys.argv[2]
    path = sys.argv[3]
    result = generate_prolog_structure(n, query, path)
    if not result:
        print("No results returned from generate_prolog_structure", file=sys.stderr)
        sys.exit(1)  # Causes subprocess to raise an error

    print(json.dumps(result))
