def letter_repeater():
    output_line = ''
    input_line = input("Enter input: ")
    while(input_line != 'stop'):
        split = input_line.split(' ')
        num, letter = int(split[0]), split[1]
        for i in range(num):
            output_line += letter
        input_line = input("Enter input: ")
    print(output_line)

if __name__ == "__main__":
    letter_repeater()