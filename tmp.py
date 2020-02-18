str = input()
if str[0] == 'C':
    C = float(str[1:])
    F = C * 1.8 + 32
    print("%.2f" %F)
else:
    F = float(str[1:])
    C = (F - 32) / 1.8
    print("%.2f" %C)