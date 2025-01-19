def calculate_output_length(length_in, kernel_size, stride=1, padding=0, dilation=1):
    return (length_in + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1



etape1 = calculate_output_length(224, 11, stride=4)
etape2 = calculate_output_length(etape1, 3,2)
etape3 = calculate_output_length(etape2, 5, padding=2)
etape4 = calculate_output_length(etape3, 3, 2)
etape5 = calculate_output_length(etape4, 3, padding=1)
etape6 = calculate_output_length(etape5, 3, padding=1)
etape7 = calculate_output_length(etape6, 3, padding=1)
etape8 = calculate_output_length(etape7, 3, 2)
# etape9 = calculate_output_length(etape8, 3, 2)
# etape10 = calculate_output_length(etape9, 3, 2)
# etape11 = calculate_output_length(etape10, 3, 2)


print(etape1)
print(etape2)
print(etape3)
print(etape4)
print(etape5)
print(etape6)
print(etape7)
print(etape8)
# print(etape9)
# print(etape10)
# print(etape11)
