import argparse


def add_numbers(a, b):
    """This function adds two numbers."""
    return a + b


def main():
    parser = argparse.ArgumentParser(
        description="A simple calculator program.")
    parser.add_argument('num1', type=int, help='The first number')
    parser.add_argument('num2', type=int, help='The second number')
    args = parser.parse_args()

    result = add_numbers(args.num1, args.num2)
    print(f'The result of {args.num1} + {args.num2} is {result}')


if __name__ == '__main__':
    main()
