import inspect
# Original ASM code
asm_code = [
    "a,b,+,=,c",
    "c,d,*,=,e",
    # ...
]

# Dictionary to keep track of the original variable names and their standardized names
var_mapping = {}

# Dictionary to keep track of each line of ASMs' position in the AST
line_positions_in_ast = {}

# Counter for the next variable name
var_counter = 1

# Standardized ASM code
standardized_asm_code = []

for line_number, line in enumerate(asm_code):
    # Split the line by commas
    tokens = line.split(',')

    # Process each token
    for i, token in enumerate(tokens):
        if token.isalpha():  # Check if the token is a variable name (you may need to modify this condition based on your specific ASM format)
            # If the variable is not already mapped, add it to the mapping and increment the counter
            if token not in var_mapping:
                var_mapping[token] = f"var{var_counter}"
                var_counter += 1

            # Substitute the original variable name with the standardized name
            tokens[i] = var_mapping[token]

    # Concatenate the tokens back into a standardized line
    standardized_line = ','.join(tokens)
    standardized_asm_code.append(standardized_line)

    # Optionally, store the line position in the AST (this would depend on how you define the positions in the AST)
    line_positions_in_ast[line_number] = 'your_ast_position_here'

# Print the standardized ASM code and mappings
print("Standardized ASM Code:")
for line in standardized_asm_code:
    print(line)

print("\nVariable Mapping:")
for original, standardized in var_mapping.items():
    print(f"{original} -> {standardized}")

print("\nLine Positions in AST:")
for line_number, position in line_positions_in_ast.items():
    print(f"Line {line_number}: {position}")
def factorial(n):
    """Calculate the factorial of a positive integer."""
    if n == 0:
        return 1
    else:
        return n * factorial(n - 1)
factorial(10)
print("file:///"+__file__+"#"+str(inspect.currentframe().f_lineno))


    
