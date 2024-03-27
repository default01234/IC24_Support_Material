# from extractTrainingDataFromJsonFile import ExtendedGroundingState, ExtendedCode, ExtendedOperand, CodeTable, ExtendedCodeTable
from extractTrainingDataFromJsonFile import *
import math
import os

# define unique_identifier_tokenizer for argName and argClass in Operand
operand_tokenizer: dict = {"": 0, None: 0}

token_serial_number = 0


def load_operand_tokenizer(tokenizer_path: str):
    global operand_tokenizer
    global token_serial_number
    if os.path.exists(tokenizer_path):
        try:
            with open(tokenizer_path, "r") as file:
                operand_tokenizer = json.load(file)
                token_serial_number = max(operand_tokenizer.values())
            return
        except Exception as e:
            print(f"An error occurred while reading the file: {e}")
            operand_tokenizer = {"": 0, None: 0}
            token_serial_number = max(operand_tokenizer.values())
            return
    else:
        operand_tokenizer = {"": 0, None: 0}
        token_serial_number = max(operand_tokenizer.values())
        return


def save_operand_tokenizer(tokenizer_path: str):
    with open(tokenizer_path, "w") as file:
        json.dump(operand_tokenizer, file, indent=4)


def update_operand_tokenizer(tokens: set):
    global operator_tokenizer
    global operand_tokenizer
    global token_serial_number
    for token_ in tokens:
        if token_ in operator_tokenizer:
            continue
        else:
            if token_ in operand_tokenizer:
                continue
            else:
                token_serial_number += 1
                operand_tokenizer[token_] = token_serial_number


# define identifier for argName in operator
operator_tokenizer = {
    "+": 1,
    "-": 2,
    "*": 3,
    "/": 4,
    "=": 5,
    "==": 6,
    "^": 7,
    "call": 8,
    ">>": 9,
    ">": 10,
    "<": 11,
    ",": 12,
    ">>": 13,
    "[]": 14,
    "LENGTH": 15,
    "ERASE": 16,
    "CLEAR": 17,
    "TYPENAME": 18,
    "&&": 19,
    "||": 20,
    "!": 21,
    "%": 22,
    "+=": 23,
    "-=": 24,
    "*=": 25,
    "/=": 26,
    "%=": 27,
    "^=": 28,
    "++": 29,
    "--": 30,
    "MAP": 31,
    "MULTIMAP": 32,
    "SET": 33,
    "MULTISET": 34,
    "TONUMB": 35,
    "TOSTRG": 36,
    "TONUM": 35,
    "TOSTRG": 36,
    "TOINT": 37,
    "NONBLOCKEXECUTE": 38,
    "BLOCKEXECUTE": 39,
    "SLEEP": 40,
    "FIND": 41,
    "COUNT": 42,
    "INSERT": 43,
    "PUSHBACK": 44,
    "PUSHFRONT": 45,
    "POPBACK": 46,
    "POPFRONT": 47,
    "BACK": 48,
    "!=": 51,
}


# Training Output Embedding
def output_embedding(extended_grounding_state: ExtendedGroundingState):
    # if extended_grounding_state.succeed:
    #     onehot_vector = [0] * (511)
    #     onehot_vector[extended_grounding_state.onehotPosition] = 1
    # else:
    #     onehot_vector = [0] * (512-1)

    # Concatenate succeed with the one-shot vector
    # return [not extended_grounding_state.succeed] + onehot_vector
    onehot_vector = [
        0
    ] * extended_grounding_state.extendedCodeTable.extendedCodeList.__len__()
    onehot_vector[extended_grounding_state.position] = 1
    return onehot_vector


def get_scientific_notation(number):
    """
    Convert a number into its scientific notation components.
    Returns:
    - significand: the coefficient of the expression (float rounded to 2 decimal places)
    - exponent: the exponent to which base 10 must be raised to produce the number
    """
    if number == 0 or number == None:
        return 0, 0

    # Using logarithm to derive the exponent
    exponent = int(math.floor(math.log10(abs(number))))

    # Derive the significand
    significand = round(number / (10**exponent), 2)

    return significand, exponent


# Operand Embedding
# operand_tokenizer for operand_embedding need to be initialized when creating extendedGroundingStateList
def extended_operand_embedding(extendedOperand: ExtendedOperand):
    """
    Convert an operand into an embedding vector.
    """
    update_operand_tokenizer(
        {extendedOperand.argName, extendedOperand.className})

    # Extracting values from the operand
    changeable = float(extendedOperand.changeable)
    isLeafNode = float(extendedOperand.isLeafNode)
    occurrenceNumber = float(extendedOperand.occurrenceNumber)
    positionInAST = extendedOperand.positionInAST + [0] * (
        8 - len(extendedOperand.positionInAST)
    )
    argType_map = {"identifier": 0, "number": 1, "string": 2, "funcName": 3}
    argType = argType_map.get(extendedOperand.argType, 0)
    argName = operand_tokenizer.get(extendedOperand.argName, 0)
    significand, exponent = get_scientific_notation(extendedOperand.argValue)
    isClass = float(extendedOperand.isClass)
    className = operand_tokenizer.get(extendedOperand.className, 0)

    return (
        [changeable, isLeafNode, occurrenceNumber]
        # + positionInAST
        + [argType, argName, significand, exponent, isClass, className]
    )


def extended_operator_embedding(extendedOperator: ExtendedOperator):
    type_mapping = {"other": 0, "call": 1, "comma": 2}

    embedded_vector = [type_mapping.get(extendedOperator.argType, 0)] + [0] * 3
    embedded_vector[
        type_mapping.get(extendedOperator.argType, 0) + 1
    ] = operator_tokenizer.get(extendedOperator.argName, 0)

    return embedded_vector


def extended_result_embedding(extendedResult: ExtendedResult):
    """
    Convert an ExtendedResult into an embedding vector.
    """
    update_operand_tokenizer(
        {extendedResult.argName, extendedResult.className})
    # Extracting values from the extendedResult similar to operand but without "isLeafNode" and "occurrenceNumber"
    changeable = float(extendedResult.changeable)
    positionInAST = extendedResult.positionInAST + [0] * (
        8 - len(extendedResult.positionInAST)
    )
    argType_map = {"identifier": 0, "number": 1, "string": 2, "funcName": 3}
    argType = argType_map.get(extendedResult.argType, 0)
    argName = operand_tokenizer.get(extendedResult.argName, 0)
    significand, exponent = get_scientific_notation(extendedResult.argValue)
    isClass = float(extendedResult.isClass)
    className = operand_tokenizer.get(extendedResult.className, 0)

    return (
        [changeable]
        # + positionInAST
        + [argType, argName, significand, exponent, isClass, className]
    )


def extended_code_embedding(extendedCode: ExtendedCode):
    if isinstance(extendedCode, ExtendedCode):
        return (
            [int(extendedCode.grounded)]
            + extended_operand_embedding(extendedCode.extendedOperand1)
            + extended_operand_embedding(extendedCode.extendedOperand2)
            + extended_operator_embedding(extendedCode.extendedOperator)
            + extended_result_embedding(extendedCode.extendedResult)
        )
    print(
        "file:///"
        + __file__
        + "#"
        + str(inspect.currentframe().f_lineno)
        + " extended_code_embedding type err: input isn't instance of ExtendedCode"
    )
    return None


# grounding_states = initialize_from_file("trainingData.json")
# extendedGroundingState = ExtendedGroundingState(grounding_states[0])
# print(extendedGroundingState.__dict__, 4)
